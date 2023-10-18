# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName     : judge.py
# @Description  : judge实现，举办方线上评测代码
# @Time         : 2023/9/18 9:44 上午
# @Author       : JD.com
# @Software     : PyCharm

import os
import sys
import json
import codecs
import chardet
import logging.config
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable
from joblib import Parallel, delayed
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

conf = {'version': 1,
        'formatters': {'myformat1': {'class': 'logging.Formatter', 'datefmt': '%Y-%m-%d %H:%M:%S',
                                     'format': '%(asctime)s %(levelname)s - %(message)s'}},
        'handlers': {'console': {'class': 'logging.StreamHandler', 'level': 'INFO', 'formatter': 'myformat1',
                                 'stream': 'ext://sys.stdout'}},
        'loggers': {'ccf_log': {'handlers': ['console', ], 'level': 'ERROR', 'propagate': False}}}
logging.config.dictConfig(conf)
logger = logging.getLogger('ccf_log')


class EnumVal(Enum):
    """
    枚举类
    """
    FRONT = '前'  # 前场
    BACK = '后'  # 后场
    ALL = '总'  # 总库存
    ONLINE = '线上'  # 线上订单
    OFFLINE = '线下'  # 线下订单，原始输入时2代表线下订单，其余数字代表线上

    ONLY_BACK_FULFILL = 'b'  # 仅后场履约
    ONLY_FRONT_FULFILL = 'f'  # 仅前场履约
    FRONT_BACK_COMB_FULFILL = 'c'  # 前后场组合履约

    INPUT_OFFLINE_CHANNEL = 2  # 输入的订单channel取值


class Param(object):
    """
    参数类
    """
    neg_inf = -99999999999  # 不能用np.inf，防止下游解析出问题
    front_stock_tol = 1e-4  # 前场库存容差
    kpi_ndigits = 4  # kpi小数点有效位数
    result_ndigits = 2  # 提交结果小数点有效位数
    kpi_columns = ['store_id', 'date', '订单履约率A', '净利润S', '履约效率D', '销售利润R', '缺货机会成本L', '损耗成本C', '履约成本V', \
                   '搬运成本V1', '拣货成本V2', '可履约订单量n_F', '可履约线上订单量n_F_online', '不可履约订单量n_N', '后场完单量n_B', \
                   '后场拣货种类k_bp', '后场拣货数量c_bp', '前场拣货种类k_fp', '前场拣货数量c_fp', '后场向前场搬货种类k_bfc', '后场向前场搬货数量c_bfc']

    result_columns = ['date', 'store_id', 'sku_id', 'x_k', 'x_m']

    back_sku_stock_cnt_ratio_upper = 0.4  # 后场备货商品数量占门店全部备货量的比例 <= 0.4
    back_sku_kind_cnt_ratio_upper = 0.2  # 后场备货商品种类占门店全部商品种类的比例 <= 0.2
    avg_fulfill_ratio_lower = 0.75  # 平均订单履约满足率
    wastage_ratio = 0.3  # 报损比例0.3，门店j，第t日前后场当天剩余未售出的商品数量的30%按照进货成本价全部报损，其他剩余的库存直接清零处理。
    A = 'A'  # A榜
    B = 'B'  # B榜

    @staticmethod
    def get_carry_cnt(cur_back_stock: float, cur_front_stock: float, quantity: float) -> float:
        """
        根据sku销量和前后场库存计算搬运量
        Args:
            cur_back_stock: 后场库存
            cur_front_stock: 前场库存
            quantity: 销量

        Returns: 搬运量

        """
        # 分段函数
        carry_cnt = 0
        if cur_back_stock <= 5:
            carry_cnt = cur_back_stock
        elif 5 < cur_back_stock <= 25:
            carry_cnt = 5 + quantity - cur_front_stock
        elif 25 < cur_back_stock <= 50:
            carry_cnt = 10 + quantity - cur_front_stock
        elif cur_back_stock > 50:
            carry_cnt = 20 + quantity - cur_front_stock
        carry_cnt = min(carry_cnt, cur_back_stock)
        return carry_cnt

    @staticmethod
    def get_ord_pickup_cost(front_pick_sku_kind: int, back_pick_sku_kind: int) -> float:
        """
        计算拣货成本
        Args:
            front_pick_sku_kind: 前场拣货sku种类
            back_pick_sku_kind: 后场拣货sku种类

        Returns: 拣货成本

        """
        cost = 0
        if front_pick_sku_kind > 0:  # 前场
            cost += (0.8 + 0.4 * front_pick_sku_kind)
        if back_pick_sku_kind > 0:  # 后场
            cost += (0.5 + 0.2 * back_pick_sku_kind)
        return cost

    @staticmethod
    def get_ord_carry_cost(back_to_front_carry_kind: int) -> float:
        """
        计算搬运成本
        Args:
            back_to_front_carry_sku_kind: 后场向前场搬运sku种类

        Returns: 拣货成本

        """
        cost = (5 * back_to_front_carry_kind)
        return cost


class Problem(object):
    """
    问题输入类
    """

    def __init__(self, order_df: pd.DataFrame, result_df: pd.DataFrame, price_df: pd.DataFrame, **kwargs):
        self.order_df = order_df
        self.result_df = result_df
        self.price_df = price_df


class Solution(object):
    """
    备货量解类
    """

    def __init__(self, problem: Problem, **kwargs):
        self.problem = problem  # 问题输入
        self.__index_to_x = dict()  # 备货量结果。如：{(商品i,门店j,前场EnumVal.FRONT,日期t): 10, ...}

    def set(self, index: tuple, x: float):
        """
        写入解
        Args:
            index: 解的索引。如：(i,j,k,t)代表(商品i,门店j,前场k,第t日)
            x: 备货量。

        Returns:

        """
        self.__index_to_x[index] = x

    def get(self, index: tuple) -> float:
        """
        读入解
        Args:
            index: 解的索引。如：(i,j,k,t)代表(商品i,门店j,前场k,第t日)
            x: 备货量。

        Returns: 备货量

        """
        if index not in self.__index_to_x.keys():
            logger.warning("[解操作] 索引(商品{}, 门店{}, 前/后场{}, 日期{})不存在。".format(index[0], index[1], index[2], index[3]))

        return self.__index_to_x.get(index)

    def get_all_stocks(self) -> dict:
        """
        读取所有天所有门店总库存
        Returns:

        """
        return self.__index_to_x

    def get_stocks(self, sku: int, store: int, dt: str) -> (float, float, float):
        """
        读取总库存、前场库存、后场库存
        Args:
            sku: 商品
            store: 门店
            dt: 日期

        Returns: 总库存、前场库存、后场库存

        """
        cur_stock = self.get(index=(sku, store, EnumVal.ALL, dt))  # 总库存
        cur_front_stock = self.get(index=(sku, store, EnumVal.FRONT, dt))  # 当前前场库存
        cur_back_stock = self.get(index=(sku, store, EnumVal.BACK, dt))  # 当前后场库存
        return cur_stock, cur_front_stock, cur_back_stock

    def add(self, index: tuple, delta_x: float):
        """
        解增加更新
        Args:
            index: 解的索引。如：(i,j,k,t)代表(商品i,门店j,前场k,第t日)
            delta_x: 增加备货量。

        Returns:

        """
        try:
            self.__index_to_x[index] += delta_x
        except:
            logger.warning("[解操作] 索引(商品{}, 门店{}, 前/后场{}, 日期{})不存在。".format(index[0], index[1], index[2], index[3]))

    def sub(self, index: tuple, delta_x: float):
        """
        解减少更新
        Args:
            index: 解的索引。如：(i,j,k,t)代表(商品i,门店j,前场k,第t日)
            delta_x: 增加备货量。

        Returns:

        """
        try:
            if self.__index_to_x[index] < delta_x:
                logger.warning(
                    "[解操作] 商品{}, 门店{}, 前/后场{}, 日期{}库存不足，缺少量为:{}".format(index[0], index[1], index[2], index[3],
                                                                        self.__index_to_x[index] - delta_x))

            self.__index_to_x[index] -= delta_x
        except:
            logger.warning("[解操作] 索引(商品{}, 门店{}, 前/后场{}, 日期{})不存在。".format(index[0], index[1], index[2], index[3]))

    def show(self):
        """
        打印
        Returns:

        """
        print(self.__index_to_x)


class DataParser(object):
    """
    数据解析类类
    """

    @staticmethod
    def load_data(order_path: str, result_path: str, price_path: str, sep=',', encoding='utf8') -> (
            pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        加载数据
        Args:
            order_path: 订单文件路径
            result_path: 提交结果文件路径
            price_path: sku价格文件路径
            sep: csv分隔符
            encoding: csv编码方式

        Returns: range_df, order_df, results_df, price_df

        """
        order_df = pd.read_csv(order_path, sep=sep, encoding=encoding)
        result_df = pd.read_csv(result_path, sep=sep, encoding=encoding)
        price_df = pd.read_csv(price_path, sep=sep, encoding=encoding)
        return order_df, result_df, price_df

    @staticmethod
    def parser_data(order_df: pd.DataFrame, result_df: pd.DataFrame, price_df: pd.DataFrame, ab: str) -> Problem:
        """
        解析预测范围信息
        Args:
            order_df: 订单df
            result_df: 提交结果df
            price_df: 价格df
            ab: 是否AB

        Returns:

        """

        # 订单表dt
        order_df['date'] = order_df.apply(func=lambda x: x['order_time'][:10], axis=1)

        # 关联价格
        order_df = pd.merge(left=order_df, right=price_df, how='left', on=['store_id', 'date', 'sku_id'])
        result_df = pd.merge(left=result_df, right=price_df, how='left', on=['store_id', 'date', 'sku_id'])

        # 取AB榜。A只取'ab_type' = 'A'的，B榜取全部：即'A' + 'B'
        if ab == Param.A:
            order_df = order_df[order_df['ab_type'] == Param.A]
            result_df = result_df[result_df['ab_type'] == Param.A]
            price_df = price_df[price_df['ab_type'] == Param.A]

        # 结果四舍五入保留2为小数
        result_df['x_k'] = result_df.apply(func=lambda x: round(x['x_k'], Param.result_ndigits), axis=1)
        result_df['x_m'] = result_df.apply(func=lambda x: round(x['x_m'], Param.result_ndigits), axis=1)

        # 不可售置为0
        result_df.loc[result_df['salable_status'] == 0, 'x_k'] = 0
        result_df.loc[result_df['salable_status'] == 0, 'x_m'] = 0

        # 前场备货量 + 一个极小数，防止前场正好卖光产生搬运。报损时扣减
        result_df['x_k'] = result_df['x_k'] + Param.front_stock_tol

        # 存储至problem
        problem = Problem(order_df=order_df, result_df=result_df, price_df=price_df)

        logger.info('[数据处理] 数据加载处理完毕')
        return problem


class ResultChecker(object):
    """
    规则检查器
    """

    @staticmethod
    def check_result(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查数据
        Args:
            solution: 解对象

        Returns:

        """
        checker_list = []
        checker_list.append(ResultChecker.check_result_row_cnt)  # 行数
        checker_list.append(ResultChecker.check_result_col_cnt)  # 列数
        checker_list.append(ResultChecker.check_result_col_name)  # 列名
        checker_list.append(ResultChecker.check_result_key)  # 主键正确性
        checker_list.append(ResultChecker.check_result_null)  # 空值
        checker_list.append(ResultChecker.check_result_repeat)  # 主键重复
        checker_list.append(ResultChecker.check_result_positive)  # 负值
        checker_list.append(ResultChecker.check_front_stock)  # 前场陈列
        checker_list.append(ResultChecker.check_back_stock)  # 后场库存

        flag, err_msg = True, ""
        for checker in checker_list:
            flag, err_msg = checker(result_df, price_df)
            if flag == False:
                return flag, err_msg
        return flag, err_msg

    @staticmethod
    def check_result_row_cnt(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查提交结果行数
        Args:
            solution: 解对象

        Returns:

        """
        err_msg = ""
        if len(result_df) != len(price_df):
            err_msg = "[规则检查] 提交结果行数不符合要求，提交结果行数:{}, 要求行数:{}！".format(len(result_df), len(price_df))
            logger.warning(err_msg)
            return False, err_msg
        return True, err_msg

    @staticmethod
    def check_result_col_cnt(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查提交结果列数
        Args:
            solution: 解对象

        Returns:

        """
        err_msg = ""
        if len(result_df.columns) != len(Param.result_columns):
            err_msg = "[规则检查] 提交结果列数不符合要求，提交结果列数:{}, 要求行数:{}！".format(len(result_df.columns), len(Param.result_columns))
            logger.warning(err_msg)
            return False, err_msg
        return True, err_msg

    @staticmethod
    def check_result_col_name(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查提交结果列名
        Args:
            solution: 解对象

        Returns:

        """
        err_msg = ""
        for col in result_df.columns:
            if col not in Param.result_columns:
                err_msg = "[规则检查] 提交结果列名不符合要求，错误列名:{}, 要求列名:{}！".format(col, Param.result_columns)
                logger.warning(err_msg)
                return False, err_msg
        return True, err_msg

    @staticmethod
    def check_result_key(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查提交结果主键
        Args:
            solution: 解对象

        Returns:

        """
        err_msg = ""
        check_df = result_df[['store_id', 'date', 'sku_id']]
        key_df = price_df[['store_id', 'date', 'sku_id']]
        key_df['flag'] = 1
        check_df = pd.merge(left=check_df, right=key_df, how='left', on=['store_id', 'date', 'sku_id'])
        if check_df.isnull().values.any():
            err_msg = "[规则检查] 提交结果数据中存在主键错误，请检查主键(date, store_id, sku_id)!"
            logger.warning(err_msg)
            return False, err_msg
        return True, err_msg

    @staticmethod
    def check_result_repeat(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查提交结果是否有重复
        Args:
            solution: 解对象

        Returns:

        """
        err_msg = ""
        check_df = result_df[['date', 'store_id', 'sku_id']]
        duplicate_cnt = check_df.duplicated().sum()
        if duplicate_cnt > 0:
            err_msg = "[规则检查] 提交结果数据中主键(date, store_id, sku_id)有重复，重复行数:{}！".format(duplicate_cnt)
            logger.warning(err_msg)
            return False, err_msg
        return True, err_msg

    @staticmethod
    def check_result_null(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查提交结果是否有空值
        Args:
            solution: 解对象

        Returns:

        """
        err_msg = ""
        if result_df.isnull().values.any():
            err_msg = "[规则检查] 提交结果数据中存在空值!"
            logger.warning(err_msg)
            return False, err_msg
        return True, err_msg

    @staticmethod
    def check_result_positive(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查结果是否大于0
        Args:
            solution: 解对象

        Returns:

        """
        err_msg = ""
        illegal_df = result_df[(result_df['x_k'] < 0) | (result_df['x_m'] < 0)]
        if len(illegal_df) > 0:
            for row_id, illegal_obj in illegal_df.iterrows():
                sku, store, date = illegal_obj.sku_id, illegal_obj.store_id, illegal_obj.date
                x_k, x_m = illegal_obj.x_k, illegal_obj.x_m
                err_msg = "[规则检查] 提交结果数据存在负值！如：[商品{},门店{},日期{}][备货数量:后{},前{}]".format(sku, store, date,
                                                                                      x_m, x_k)
                logger.warning(err_msg)
                return False, err_msg

        return True, err_msg

    @staticmethod
    def check_front_stock(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查前场库存
        Args:
            solution: 解对象

        Returns:

        """
        err_msg = ""
        illegal_df = result_df[(result_df['x_k'] <= 0) & (result_df['x_m'] > 0)]
        if len(illegal_df) > 0:
            for row_id, illegal_obj in illegal_df.iterrows():
                sku, store, date = illegal_obj.sku_id, illegal_obj.store_id, illegal_obj.date
                x_k, x_m = illegal_obj.x_k, illegal_obj.x_m
                err_msg = "[规则检查] 前场陈列不符合赛题要求！如：[商品{},门店{},日期{}][备货数量:后{}>0, 前{}=0]".format(sku, store, date,
                                                                                            x_m, x_k)
                logger.warning(err_msg)
                return False, err_msg

        return True, err_msg

    @staticmethod
    def check_back_stock(result_df: pd.DataFrame, price_df: pd.DataFrame) -> (bool, str):
        """
        检查后场库存
        Args:
            solution: 解对象

        Returns:

        """

        err_msg = ""
        kind_ratio = Param.back_sku_kind_cnt_ratio_upper
        stock_ratio = Param.back_sku_stock_cnt_ratio_upper

        # 总备货种类和数量，后场备货数量
        stock_df = result_df[['date', 'store_id', 'sku_id', 'x_k', 'x_m']].groupby(
            ['date', 'store_id']).agg({'sku_id': 'nunique', 'x_k': 'sum', 'x_m': 'sum'}).reset_index()
        stock_df = stock_df.rename(columns={'sku_id': 'kind_cnt', 'x_k': 'front_stock_cnt',
                                            'x_m': 'back_stock_cnt'})
        stock_df['stock_cnt'] = stock_df['front_stock_cnt'] + stock_df['back_stock_cnt']

        # 后场最多摆放sku种类和数量，参赛选手会输出所有sku，即使补货为0
        stock_df['max_back_sku_kind_cnt'] = stock_df['kind_cnt'] * kind_ratio
        stock_df['max_back_sku_stock_cnt'] = stock_df['stock_cnt'] * stock_ratio

        # 后场备货种类
        back_stock_df = result_df[result_df['x_m'] > 0]
        back_stock_df = back_stock_df[['date', 'store_id', 'sku_id']].groupby(['date', 'store_id']).agg(
            {'sku_id': 'nunique'}).reset_index()
        back_stock_df = back_stock_df.rename(columns={'sku_id': 'back_kind_cnt'})

        # 合并
        stock_df = pd.merge(left=stock_df, right=back_stock_df, how='left', on=['date', 'store_id'])

        # 检查
        stock_df['back_sku_kind_cnt_diff'] = stock_df['max_back_sku_kind_cnt'] - stock_df['back_kind_cnt']
        stock_df['back_sku_stock_cnt_diff'] = stock_df['max_back_sku_stock_cnt'] - stock_df['back_stock_cnt']

        # 筛选出不合法（date, store）
        illegal_df = stock_df[
            (stock_df['back_sku_kind_cnt_diff'] < 0) | (stock_df['back_sku_stock_cnt_diff'] < 0)]

        if len(illegal_df) > 0:
            for row_id, illegal_obj in illegal_df.iterrows():
                store, date = illegal_obj.store_id, illegal_obj.date
                back_kind_cnt, back_stock_cnt = illegal_obj.back_kind_cnt, illegal_obj.back_stock_cnt
                kind_cnt, stock_cnt = illegal_obj.kind_cnt, illegal_obj.stock_cnt
                if back_kind_cnt > kind_cnt * kind_ratio:
                    err_msg = "[规则检查] 后场库存不符合赛题要求！如：[门店{},日期{},后场备货种类{}超过总备货种类{}*{}={}]" \
                        .format(store, date, back_kind_cnt, kind_cnt, kind_ratio,
                                round(kind_cnt * kind_ratio, Param.kpi_ndigits))

                    logger.warning(err_msg)
                    return False, err_msg

                if back_stock_cnt > stock_cnt * stock_ratio:
                    err_msg = "[规则检查] 后场库存不符合赛题要求！如：[门店{},日期{},后场备货数量{}超过总备货数量{}*{}={}]" \
                        .format(store, date, round(back_stock_cnt, Param.kpi_ndigits),
                                round(stock_cnt, Param.kpi_ndigits), stock_ratio,
                                round(stock_cnt * stock_ratio, Param.kpi_ndigits))
                    logger.warning(err_msg)
                    return False, err_msg

                err_msg = "[规则检查] 后场库存不符合赛题要求！"
                logger.warning(err_msg)
                return False, err_msg
        return True, err_msg


class Helper(object):
    """
    计算辅助类
    """

    @staticmethod
    def check_is_ord_fulfill(solution: Solution, order_detail_df: pd.DataFrame) -> bool:
        """
        检查订单全库存是否可履约
        Args:
            solution: 解对象
            order_detail_df: 订单信息

        Returns: 是否可履约

        """
        for _, ord in order_detail_df.iterrows():
            # 库存不足不能履约
            front_stock = solution.get(index=(ord.sku_id, ord.store_id, EnumVal.FRONT, ord.date))  # 前场库存
            back_stock = solution.get(index=(ord.sku_id, ord.store_id, EnumVal.BACK, ord.date))  # 后场库存
            total_stock = front_stock + back_stock
            if ord.quantity > total_stock:
                logger.warning('[规则检查] [库存不足][商品{},门店{},日期{},订单{},销量{},前场库存{},后场库存{},总库存{}]' \
                               .format(ord.sku_id, ord.store_id, ord.date, ord.order_id, ord.quantity,
                                       front_stock, back_stock, total_stock))
                return False
        return True

    @staticmethod
    def get_carry_cnt(cur_back_stock: float, cur_front_stock: float, quantity: float) -> float:
        """
        根据sku销量和前后场库存每次搬运量
        Args:
            cur_back_stock: 后场库存
            cur_front_stock: 前场库存
            quantity: 销量

        Returns: 搬运量

        """
        return Param.get_carry_cnt(cur_back_stock, cur_front_stock, quantity)

    @staticmethod
    def get_ord_carry_cost(back_to_front_carry_kind: int) -> float:
        """
        计算订单成本
        Args:
            back_to_front_carry_sku_kind: 后场向前场搬运sku种类

        Returns: 拣货成本

        """
        return Param.get_ord_carry_cost(back_to_front_carry_kind)

    @staticmethod
    def get_ord_pickup_cost(front_pick_sku_kind: int, back_pick_sku_kind: int) -> float:
        """
        计算订单拣货成本
        Args:
            front_pick_sku_kind: 前场拣货sku种类
            back_pick_sku_kind: 后场拣货sku种类

        Returns: 拣货成本

        """
        return Param.get_ord_pickup_cost(front_pick_sku_kind, back_pick_sku_kind)

    @staticmethod
    def get_ord_sale_profit(order_detail_df: pd.DataFrame) -> float:
        """
        计算订单销售利润
        Args:
            F_jt: 某个门店某一天的可履约订单

        Returns: 缺货机会成本

        """
        sale_profit = 0
        if len(order_detail_df) > 0:
            sale_profit_df = order_detail_df.copy()
            sale_profit_df['profit'] = sale_profit_df['sale_price'] - sale_profit_df['cost_price']
            sale_profit_df['sale_profit'] = sale_profit_df['profit'] * sale_profit_df['quantity']
            sale_profit = float(sale_profit_df['sale_profit'].sum())
        return sale_profit

    @staticmethod
    def get_ord_stockout_cost(order_detail_df: pd.DataFrame) -> float:
        """
        计算订单缺货机会成本
        Args:
            N_jt: 某个门店某一天的不可履约订单

        Returns: 缺货机会成本

        """
        stockout_cost = 0
        if len(order_detail_df) > 0:
            stockout_ord_df = order_detail_df.copy()
            stockout_ord_df['profit'] = stockout_ord_df['sale_price'] - stockout_ord_df['cost_price']
            stockout_ord_df['stockout_cost'] = stockout_ord_df['profit'] * stockout_ord_df['quantity']
            stockout_cost = float(stockout_ord_df['stockout_cost'].sum())
        return stockout_cost

    @staticmethod
    def get_store_dt_wastage_cost(solution: Solution, store: int, dt: str) -> float:
        """
        计算某个门店某一天的损耗成本
        Args:
            solution: 解对象
            store: 门店j
            dt: 日期t

        Returns: 缺货机会成本

        """
        result_df = solution.problem.result_df
        wastage_df = result_df[(result_df['store_id'] == store) & (result_df['date'] == dt)]
        sku_wastage_cnts = []
        for sku in wastage_df['sku_id'].values.tolist():
            back_wastage_cnt = solution.get(index=(sku, store, EnumVal.BACK, dt))
            # 扣减防止前场正好卖光产生搬运而微调的量
            front_wastage_cnt = solution.get(index=(sku, store, EnumVal.FRONT, dt)) - Param.front_stock_tol
            wastage_cnt = (back_wastage_cnt + front_wastage_cnt) * Param.wastage_ratio  # 30% 报损
            sku_wastage_cnts.append(wastage_cnt)
        wastage_df['wastage_cnt'] = sku_wastage_cnts
        wastage_df['wastage_cost'] = wastage_df['wastage_cnt'] * wastage_df['cost_price']
        wastage_cost = float(wastage_df['wastage_cost'].sum())
        return wastage_cost

    @staticmethod
    def print_final_kpi(kpi_df: pd.DataFrame):
        """
        打印final_kpi
        Args:
            final_kpi_df:

        Returns:

        """
        for row_id, kpi in kpi_df.iterrows():
            store = kpi['store_id']
            dt = kpi['date']
            A = kpi['订单履约率A']
            S = kpi['净利润S']
            D = kpi['履约效率D']
            R = kpi['销售利润R']
            L = kpi['缺货机会成本L']
            C = kpi['损耗成本C']
            V = kpi['履约成本V']
            V1 = kpi['搬运成本V1']
            V2 = kpi['拣货成本V2']
            n_F = kpi['可履约订单量n_F']
            n_F_online = kpi['可履约线上订单量n_F_online']
            n_N = kpi['不可履约订单量n_N']
            n_B = kpi['后场完单量n_B']
            k_bp = kpi['后场拣货种类k_bp']
            c_bp = kpi['后场拣货数量c_bp']
            k_fp = kpi['前场拣货种类k_fp']
            c_fp = kpi['前场拣货数量c_fp']
            k_bfc = kpi['后场向前场搬货种类k_bfc']
            c_bfc = kpi['后场向前场搬货数量c_bfc']
            logger.info(
                '[指标汇总] [门店:{},日期:{}][订单履约率A:{},净利润S:{},履约效率D:{},销售利润R:{},缺货机会成本L:{},损耗成本C:{},履约成本V:{},搬运成本V1:{},拣货成本V2:{},可履约订单量n_F:{},可履约线上订单量n_F_online:{},不可履约订单量n_N:{},后场完单量n_B:{},后场拣货种类k_bp:{},后场拣货数量c_bp:{},前场拣货种类k_fp:{},前场拣货数量c_fp:{},后场向前场搬货种类k_bfc:{},后场向前场搬货数量c_bfc:{}]'. \
                    format(store, dt, A, S, D, R, L, C, V, V1, V2, n_F, n_F_online, n_N, n_B, k_bp, c_bp, k_fp, c_fp,
                           k_bfc, c_bfc))

    @staticmethod
    def round_kpi(A, S, D, R, L, C, V, V1, V2, n_F, n_F_online, n_N, n_B, k_bp, c_bp, k_fp, c_fp, k_bfc, c_bfc):
        """
        四舍五入保留小数
        Args:
            ...: 指标

        Returns:

        """
        A = round(A, Param.kpi_ndigits)
        S = round(S, Param.kpi_ndigits)
        D = round(D, Param.kpi_ndigits)
        R = round(R, Param.kpi_ndigits)
        L = round(L, Param.kpi_ndigits)
        C = round(C, Param.kpi_ndigits)
        V = round(V, Param.kpi_ndigits)
        V1 = round(V1, Param.kpi_ndigits)
        V2 = round(V2, Param.kpi_ndigits)
        n_F = round(n_F, Param.kpi_ndigits)
        n_F_online = round(n_F_online, Param.kpi_ndigits)
        n_N = round(n_N, Param.kpi_ndigits)
        n_B = round(n_B, Param.kpi_ndigits)
        k_bp = round(k_bp, Param.kpi_ndigits)
        c_bp = round(c_bp, Param.kpi_ndigits)
        k_fp = round(k_fp, Param.kpi_ndigits)
        c_fp = round(c_fp, Param.kpi_ndigits)
        k_bfc = round(k_bfc, Param.kpi_ndigits)
        c_bfc = round(c_bfc, Param.kpi_ndigits)
        return A, S, D, R, L, C, V, V1, V2, n_F, n_F_online, n_N, n_B, k_bp, c_bp, k_fp, c_fp, k_bfc, c_bfc

    @staticmethod
    def cal_final_kpi(store_dt_kpi_list_list):
        """
        计算最终kpi
        Args:
            store_dt_kpi_list_list: 各门店日期的kpi集合

        Returns: '订单履约率A', '净利润S', '履约效率D', '销售利润R', '缺货机会成本L', '损耗成本C', '履约成本V',
                   '搬运成本V1', '拣货成本V2', '可履约订单量n_F', '可履约线上订单量n_F_online', '不可履约订单量n_N', '后场完单量n_B',
                   '后场拣货种类k_bp', '后场拣货数量c_bp', '前场拣货种类k_fp', '前场拣货数量c_fp', '后场向前场搬货种类k_bfc', '后场向前场搬货数量c_bfc'
        """
        store_dt_kpi_df = pd.DataFrame(data=store_dt_kpi_list_list, columns=Param.kpi_columns)
        store_dt_kpi_df = store_dt_kpi_df.sort_values(by=['store_id', 'date'], ascending=[True, True]).reset_index()
        store_dt_kpi_df = store_dt_kpi_df[Param.kpi_columns]
        R = store_dt_kpi_df['销售利润R'].sum()
        L = store_dt_kpi_df['缺货机会成本L'].sum()
        C = store_dt_kpi_df['损耗成本C'].sum()
        V = store_dt_kpi_df['履约成本V'].sum()
        V1 = store_dt_kpi_df['搬运成本V1'].sum()
        V2 = store_dt_kpi_df['拣货成本V2'].sum()
        n_F = store_dt_kpi_df['可履约订单量n_F'].sum()
        n_F_online = store_dt_kpi_df['可履约线上订单量n_F_online'].sum()
        n_N = store_dt_kpi_df['不可履约订单量n_N'].sum()
        n_B = store_dt_kpi_df['后场完单量n_B'].sum()
        k_bp = store_dt_kpi_df['后场拣货种类k_bp'].sum()
        c_bp = store_dt_kpi_df['后场拣货数量c_bp'].sum()
        k_fp = store_dt_kpi_df['前场拣货种类k_fp'].sum()
        c_fp = store_dt_kpi_df['前场拣货数量c_fp'].sum()
        k_bfc = store_dt_kpi_df['后场向前场搬货种类k_bfc'].sum()
        c_bfc = store_dt_kpi_df['后场向前场搬货数量c_bfc'].sum()
        S = store_dt_kpi_df['净利润S'].sum()
        A = n_F / max(1, n_F + n_N)
        D = n_B / max(1, n_F_online)
        A, S, D, R, L, C, V, V1, V2, n_F, n_F_online, n_N, n_B, k_bp, c_bp, k_fp, c_fp, k_bfc, c_bfc = Helper.round_kpi(
            A, S, D, R, L, C, V, V1, V2, n_F, n_F_online, n_N, n_B, k_bp, c_bp, k_fp, c_fp, k_bfc, c_bfc)  # 四舍五入

        return A, S, D, R, L, C, V, V1, V2, n_F, n_F_online, n_N, n_B, k_bp, c_bp, k_fp, c_fp, k_bfc, c_bfc, store_dt_kpi_df

    @staticmethod
    def copy_single_store_dt_solution_list(total_solution: Solution) -> dict:
        """
        分门店分天copy solution
        Args:
            total_solution:

        Returns:

        """
        result_df = total_solution.problem.result_df
        store_dt_to_sub_dict = {}
        for store_dt_obj, _ in result_df.groupby(['store_id', 'date']):
            store, dt = store_dt_obj[0], store_dt_obj[1]
            sub_sol = Solution(total_solution.problem)
            store_dt_to_sub_dict[(store, dt)] = sub_sol

        index_to_x = total_solution.get_all_stocks()
        for k, v in index_to_x.items():
            store, dt = k[1], k[3]  # index: sku, store, 场, dt
            sub_sol = store_dt_to_sub_dict.get((store, dt))
            sub_sol.set(index=k, x=v)
        return store_dt_to_sub_dict


class LocalParallel(object):
    """
    本地并行算法库
    """

    class ParallelUnit(object):
        """
        并行计算单元
        """

        def __init__(self, seq: int, call_func: Callable, **kwargs):
            self.seq = seq  # 输入参数序号，求解顺序
            self.call_func = call_func
            self.call_func_input: dict = {}
            for k, v in kwargs.items():
                self.call_func_input.update({k: v})
            self.call_func_output = None

    def __init__(self):
        self.parallel_unit_list = []  # 并行计算单元

    def parallel_run(self, n_jobs: int = 1) -> list:
        '''
        并行调度
        Args:
            parallel_unit_list: 并行集合
            n_jobs: 并行单元

        Returns: 结果列表

        '''
        # 求解
        ret_list = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(LocalParallel.parallel_search)(self.parallel_unit_list, i) for i in
            range(len(self.parallel_unit_list)))

        # 合并
        merge_ret_list = []
        ret_list.sort(key=lambda x: x.seq)  # 保持顺序一致
        for parallel_unit in ret_list:
            merge_ret_list.extend(parallel_unit.call_func_output)

        # 返回
        return merge_ret_list

    def series_run(self) -> list:
        '''
        串行调度
        Args:
            parallel_unit_list: 并行集合
            n_jobs: 并行单元

        Returns: 结果列表

        '''
        # 求解
        ret_list = []
        for parallel_unit in self.parallel_unit_list:
            call_func = parallel_unit.call_func
            parallel_unit.call_func_output = call_func(**parallel_unit.call_func_input)
            ret_list.append(parallel_unit)

        # 合并
        merge_ret_list = []
        ret_list.sort(key=lambda x: x.seq)  # 保持顺序一致
        for parallel_unit in ret_list:
            merge_ret_list.extend(parallel_unit.call_func_output)

        # 返回
        return merge_ret_list

    @staticmethod
    def parallel_search(parallel_unit_list, i):
        """
        并行搜索单元
        Args:
            parallel_unit_list: 并行列表
            i: 并行单元序号
        Returns: 并行单元

        """
        parallel_unit = parallel_unit_list[i]
        call_func = parallel_unit.call_func
        parallel_unit.call_func_output = call_func(**parallel_unit.call_func_input)
        return parallel_unit


class Evaluator(object):
    """
    结果评估器类
    """

    @staticmethod
    def run(order_path: str, result_path: str, price_path: str, ab: str = Param.A, n_jobs=4, to_csv=False) \
            -> (bool, str, float, float, float, str):
        """
        所有门店订单回放方法
        Args:
            order_path: 订单文件路径
            result_path: 提交结果文件路径
            price_path: sku价格文件路径
            ab：AB榜
            n_jobs: 并行数
            to_csv: 是否输出csv

        Returns: 成绩是否有效，报错信息，净利润S，订单履约率A，履约效率D，kpi_str

        """

        logger.info('[评测程序] 开始运行...')

        S, D, A, kpi_str = Param.neg_inf, Param.neg_inf, Param.neg_inf, ''

        err_msg = ""
        start_time = time.time()

        # 1.加载输入
        order_df, result_df, price_df = DataParser.load_data(order_path=order_path, result_path=result_path,
                                                             price_path=price_path)

        # 2 数据校验
        flag, err_msg = ResultChecker.check_result(result_df, price_df)
        if not flag:
            return flag, err_msg, S, D, A, kpi_str

        # 3 处理输入
        problem = DataParser.parser_data(order_df=order_df, result_df=result_df, price_df=price_df, ab=ab)

        # 4 创建解
        solution = Solution(problem)
        Evaluator.init_stock(solution)  # 初始库存

        # 5 分门店日期统计回放订单集合
        store_dt_order_df_list = []
        for store_dt_obj, store_dt_order_df in solution.problem.order_df.groupby(['store_id', 'date']):
            store, dt = store_dt_obj[0], store_dt_obj[1]
            store_dt_order_df_list.append((store, dt, store_dt_order_df))
        store_dt_order_df_list = sorted(store_dt_order_df_list, key=lambda x: (x[0], x[1]))

        # 6 订单回放
        if n_jobs == 1:
            store_dt_kpi_list_list = Evaluator.series_run(solution, store_dt_order_df_list)
        else:
            store_dt_kpi_list_list = Evaluator.parallel_run(solution, store_dt_order_df_list, n_jobs=n_jobs)

        # 7 计算最终kpi
        A, S, D, R, L, C, V, V1, V2, n_F, n_F_online, n_N, n_B, k_bp, c_bp, k_fp, c_fp, k_bfc, c_bfc, store_dt_kpi_df \
            = Helper.cal_final_kpi(store_dt_kpi_list_list)

        # # 8 校验平均订单履约率校验。备注：现在要求也展示出来
        # if A < Param.avg_fulfill_ratio_lower:
        #     S, D = Param.neg_inf, Param.neg_inf
        #     err_msg = "[规则检查] 提交结果的平均订单履约率不符合要求，提交结果平均订单履约率:{}, 要求平均订单履约率:{}!". \
        #         format(A, Param.avg_fulfill_ratio_lower)
        #     logger.warning(err_msg)
        #     return False, err_msg, S, D, A, kpi_str

        # 9 输出最终kpi
        all_kpi_list_list = [['全部门店', '全部日期', A, S, D, R, L, C, V, V1, V2, n_F, n_F_online, n_N, n_B, k_bp, \
                              c_bp, k_fp, c_fp, k_bfc, c_bfc]]
        all_kpi_df = pd.DataFrame(data=all_kpi_list_list, columns=Param.kpi_columns)
        final_kpi_df = pd.concat([all_kpi_df, store_dt_kpi_df], axis=0)

        if to_csv:
            final_kpi_df.to_csv('final_kpi_df.csv', sep=',', encoding='utf8', index=False)

        logger.info('******************************* 最终指标(显示全部小数位) *******************************')
        Helper.print_final_kpi(store_dt_kpi_df)
        Helper.print_final_kpi(all_kpi_df)

        # 拼接最后全局指标字符串
        kpi_str = ''
        for col in all_kpi_df.columns:
            val = all_kpi_df[col].values.tolist()[0]
            if not isinstance(val, str):
                val = str(val)  # 对外展示4位小数
            one_kpi_str = col + ':' + val + ','
            kpi_str += one_kpi_str

        end_time = time.time()
        run_time = round(end_time - start_time, Param.kpi_ndigits)
        kpi_str = kpi_str + '评测耗时:' + str(run_time) + '秒'

        logger.info('[评测程序] 结束运行!')
        return True, err_msg, S, D, A, kpi_str

    @staticmethod
    def parallel_run(solution: Solution, store_dt_order_df_list: list, n_jobs) -> list:
        """
        并行求解
        Args:
            solution: 解对象
            store_dt_order_df_list: 门店日期订单集合
            n_jobs: 并行数

        Returns: 门店日期kpi集合

        """
        store_dt_kpi_list_list = []
        store_dt_to_sub_dict = Helper.copy_single_store_dt_solution_list(solution)
        local_parallel = LocalParallel()
        for seq, obj in enumerate(store_dt_order_df_list):
            store, dt, store_dt_order_df = obj[0], obj[1], obj[2]
            single_store_dt_solution = store_dt_to_sub_dict.get((store, dt))
            parallel_unit = LocalParallel.ParallelUnit(seq=seq, call_func=Evaluator.single_store_dt_run,
                                                       solution=single_store_dt_solution, store=store, dt=dt,
                                                       store_dt_order_df=store_dt_order_df)
            local_parallel.parallel_unit_list.append(parallel_unit)
        merge_ret_list = local_parallel.parallel_run(n_jobs=n_jobs)
        for kpi_list in merge_ret_list:
            store_dt_kpi_list_list.append(kpi_list)
        return store_dt_kpi_list_list

    @staticmethod
    def series_run(solution: Solution, store_dt_order_df_list: list) -> list:
        """
        并行求解
        Args:
            solution: 解对象
            store_dt_order_df_list: 门店日期订单集合

        Returns: 门店日期kpi集合

        """
        store_dt_kpi_list_list = []
        for store, dt, store_dt_order_df in store_dt_order_df_list:
            kpi_list_list = Evaluator.single_store_dt_run(solution, store, dt, store_dt_order_df)
            if len(kpi_list_list) > 0:
                store_dt_kpi_list_list.append(kpi_list_list[0])
        return store_dt_kpi_list_list

    @staticmethod
    def single_store_dt_run(solution: Solution, store: int, dt: str, store_dt_order_df: pd.DataFrame) -> list:
        """
        单个门店订单回放方法
        Args:
            solution: 解对象
            store: 门店j
            dt: 日期t
            single_store_order_df: 单个门店单个日期订单数据

        Returns: kpi_list

        """
        # log
        logger.info('[订单回放-门店日期] [门店{},日期{}] 开始回放...'.format(store, dt))

        # 初始化集合和数值：j代表某个store，t代表dt
        F_jt, F_online_jt, N_jt, B_jt = list(), list(), list(), list()
        V1_jt, V2_jt, V_jt, R_jt, L_jt, C_jt, S_jt = 0, 0, 0, 0, 0, 0, 0
        all_back_pick_cnt, all_front_pick_cnt, all_back_to_front_carry_cnt = 0, 0, 0
        all_back_pick_kind, all_front_pick_kind, all_back_to_front_carry_kind = 0, 0, 0

        # 订单排序回放
        # 1 创建订单回放集合
        single_order_infos = Evaluator.create_playback_ords(store_dt_order_df)

        # 2 排序。订单时间从小到大，订单时间相同订单号从小到大
        sorted_single_order_infos = sorted(single_order_infos, key=lambda x: (x[0], x[1]))

        # 3 逐单回放
        for ord_info in sorted_single_order_infos:
            order_time, order_id, order_type, order_detail_df = ord_info[0], ord_info[1], ord_info[2], ord_info[3]

            # 3.1 单个订单回放
            is_ord_fulfill, is_online_ord_fulfill, is_ord_only_back_fulfill, sale_profit, stockout_cost, carry_cost, \
            pickup_cost, fulfill_cost, ord_back_pick_kind, ord_front_pick_kind, ord_back_to_front_carry_kind, \
            ord_back_pick_cnt, ord_front_pick_cnt, ord_back_to_front_carry_cnt \
                = Evaluator.playback_order(solution, store, dt, ord_info)

            # 3.2 更新指标
            # 3.2.1 可履约指标
            if is_ord_fulfill:
                # 可履约订单
                F_jt.append(order_detail_df)  # 订单添加进F_jt
                # 线上可履约订单
                if is_online_ord_fulfill:
                    F_online_jt.append(order_detail_df)  # 订单添加进F_online_jt
                # 后场完单
                if is_ord_only_back_fulfill:
                    B_jt.append(order_detail_df)  # 订单添加进后场B_jt
                # 销售利润
                R_jt += sale_profit  # 更新R_jt
                # 搬运成本
                V1_jt += carry_cost  # 更新V1_jt
                # 拣货成本
                V2_jt += pickup_cost  # 更新V2_jt
                # 履约成本
                V_jt += fulfill_cost  # 更新V_jt

            # 3.2.2 不可履约指标
            else:
                # 不可履约订单
                N_jt.append(order_detail_df)  # 订单添加进N_jt
                # 缺货成本
                L_jt += stockout_cost  # 更新L_jt

            # 3.2.3 拣货/搬运种类和数量
            all_back_pick_cnt += ord_back_pick_cnt
            all_front_pick_cnt += ord_front_pick_cnt
            all_back_to_front_carry_cnt += ord_back_to_front_carry_cnt
            all_back_pick_kind += ord_back_pick_kind
            all_front_pick_kind += ord_front_pick_kind
            all_back_to_front_carry_kind += ord_back_to_front_carry_kind

            # log
            logger.info(
                '[回放细节-订单] [订单{},订单类型{}][可履约{},后场完单{}][拣货种类:前{},后{}][搬运种类{}][销售利润{},搬运成本{},拣货成本{},履约成本{},缺货成本{}][累计销售利润R:{}, 累计搬运成本V1:{}, 累计拣货成本V2:{}, 累计履约成本V:{}, 累计缺货成本L:{}][累计可履约订单量n_F:{}, 累计线上可履约订单量n_F_online:{}, 累计不可履约订单量n_N:{}, 累计后场完单量n_B:{}]'. \
                    format(order_id, order_type.value, is_ord_fulfill, is_ord_only_back_fulfill, \
                           ord_front_pick_kind, ord_back_pick_kind, ord_back_to_front_carry_kind, \
                           round(sale_profit, Param.kpi_ndigits), round(carry_cost, Param.kpi_ndigits),
                           round(pickup_cost, Param.kpi_ndigits),
                           round(fulfill_cost, Param.kpi_ndigits), round(stockout_cost, Param.kpi_ndigits), \
                           round(R_jt, Param.kpi_ndigits), round(V1_jt, Param.kpi_ndigits),
                           round(V2_jt, Param.kpi_ndigits), \
                           round(V_jt, Param.kpi_ndigits), round(L_jt, Param.kpi_ndigits), len(F_jt), len(F_online_jt),
                           len(N_jt), len(B_jt)))

            # log
            logger.info('[订单回放] [订单{},订单时间{},订单类型:{}] 回放完成'. \
                        format(store, dt, order_id, order_time, order_type.value))
            logger.info('---------------------------------------------------------')
            # dbg 已校验，拣货和搬运种类和数量正确
            # if ord_front_pick_kind == 0 and ord_back_pick_kind == 0:
            #     raise(1)
            # if ord_front_pick_kind > 0 and ord_back_pick_kind == 0:
            #     raise(1)
            # if ord_front_pick_kind > 0 and ord_back_pick_kind > 0:
            #     raise(1)
            # if ord_front_pick_kind == 0 and ord_back_pick_kind > 0:
            #     raise(1)
            # if ord_back_to_front_carry_kind > 0:
            #     raise(1)

        # 损耗成本
        C_jt = Helper.get_store_dt_wastage_cost(solution, store, dt)
        # 净利润
        S_jt = R_jt - L_jt - C_jt - V_jt
        # 订单履约率
        A_jt = len(F_jt) / max(1, len(F_jt) + len(N_jt))
        # 履约效率
        D_jt = len(B_jt) / max(1, len(F_online_jt))
        # kpi_list
        kpi_list = [store, dt, A_jt, S_jt, D_jt, R_jt, L_jt, C_jt, V_jt, V1_jt, V2_jt, len(F_jt), len(F_online_jt),
                    len(N_jt), len(B_jt), all_front_pick_kind, all_front_pick_cnt, all_back_pick_kind,
                    all_back_pick_cnt, all_back_to_front_carry_kind, all_back_to_front_carry_cnt]

        logger.info(
            '[回放细节-门店日期] [门店{},日期{}][订单履约率A:{},净利润S:{},履约效率D:{},销售利润R:{},缺货机会成本L:{},损耗成本C:{},履约成本V:{},搬运成本V1:{},拣货成本V2:{},可履约订单量n_F:{},可履约线上订单量n_F_online:{},不可履约订单量n_N:{},后场完单量n_B:{},后场拣货种类k_bp:{},后场拣货数量c_bp:{},前场拣货种类k_fp:{},前场拣货数量c_fp:{},后场向前场搬货种类k_bfc:{},后场向前场搬货数量c_bfc:{}]'. \
                format(store, dt, round(A_jt, Param.kpi_ndigits), round(S_jt, Param.kpi_ndigits),
                       round(D_jt, Param.kpi_ndigits),
                       round(R_jt, Param.kpi_ndigits), round(L_jt, Param.kpi_ndigits), round(C_jt, Param.kpi_ndigits),
                       round(V_jt, Param.kpi_ndigits), round(V1_jt, Param.kpi_ndigits), round(V2_jt, Param.kpi_ndigits),
                       len(F_jt),
                       len(F_online_jt), len(N_jt), len(B_jt), all_front_pick_kind,
                       round(all_front_pick_cnt, Param.kpi_ndigits), all_back_pick_kind,
                       round(all_back_pick_cnt, Param.kpi_ndigits), all_back_to_front_carry_kind,
                       round(all_back_to_front_carry_cnt, Param.kpi_ndigits)))

        logger.info('[订单回放-门店日期] [门店{},日期{}] 回放完成'.format(store, dt))
        return [kpi_list]  # 为了适配多线程

    @staticmethod
    def init_stock(solution: Solution):
        """
        初始化库存
        Args:
            solution: 解对象

        Returns:

        """
        for row_id, x_obj in solution.problem.result_df.iterrows():
            solution.set(index=(x_obj.sku_id, x_obj.store_id, EnumVal.FRONT, x_obj.date), x=x_obj.x_k)  # 前场备货
            solution.set(index=(x_obj.sku_id, x_obj.store_id, EnumVal.BACK, x_obj.date), x=x_obj.x_m)  # 后场备货
            solution.set(index=(x_obj.sku_id, x_obj.store_id, EnumVal.ALL, x_obj.date), x=x_obj.x_k + x_obj.x_m)  # 总库存

    @staticmethod
    def create_playback_ords(store_dt_order_df: pd.DataFrame) -> list:
        """
        创建某个门店某一天的订单回放集合
        Args:
            store_dt_order_df: 某个门店某一天的订单全集

        Returns: 订单集合

        """
        # 创建订单回放集合
        single_order_infos = []  # 订单集合
        for order_info, order_detail_df in store_dt_order_df.groupby(['order_time', 'order_id']):
            order_time, order_id = order_info[0], order_info[1]
            order_type = EnumVal.ONLINE
            if int(order_detail_df['channel'].max()) == EnumVal.INPUT_OFFLINE_CHANNEL.value:  # 每个订单只有一个渠道
                order_type = EnumVal.OFFLINE
            single_order_infos.append((order_time, order_id, order_type, order_detail_df))
        return single_order_infos

    @staticmethod
    def playback_order(solution: Solution, store: int, dt: str, ord_info: tuple):
        """
        回放订单
        Args:
            solution: 解对象
            store: 门店j
            dt: 日期t
            ord_info: 订单元组

        Returns:
            is_ord_fulfill: 是否可履约订单
            is_online_ord_fulfill: 是否线可履约线上订单
            is_ord_only_back_fulfill: 是否后场完单
            sale_profit: 销售利润
            stockout_cost: 缺货机会成本
            carry_cost: 搬运成本
            pickup_cost: 拣货成本
            fulfill_cost: 履约成本
            ord_back_pick_kind: 订单后场拣货种类
            ord_front_pick_kind: 订单前场拣货种类
            ord_back_to_front_carry_kind: 订单后场向前场搬运种类
            ord_back_pick_cnt: 订单后场拣货数量
            ord_front_pick_cnt: 订单前场拣货数量
            ord_back_to_front_carry_cnt: 订单后场向前场搬运数量

        """

        order_time, order_id, order_type, order_detail_df = ord_info[0], ord_info[1], ord_info[2], ord_info[3]
        # log
        logger.info('[订单回放-订单] [订单{},订单时间{},订单类型:{}] 开始回放...'.format(order_id, order_time, order_type.value))
        ord_back_pick_kind, ord_front_pick_kind, ord_back_to_front_carry_kind = 0, 0, 0
        ord_back_pick_cnt, ord_front_pick_cnt, ord_back_to_front_carry_cnt = 0, 0, 0
        sale_profit, stockout_cost, carry_cost, pickup_cost, fulfill_cost = 0, 0, 0, 0, 0
        is_ord_fulfill, is_online_ord_fulfill, is_ord_only_back_fulfill = False, False, False

        # 先判定订单是否能履约。只有所有sku均库存充足时才能履约
        is_ord_fulfill = Helper.check_is_ord_fulfill(solution, order_detail_df)
        # 不可履约
        if not is_ord_fulfill:
            is_ord_fulfill = False
        # 可履约
        else:
            is_ord_fulfill = True

            # 可履约且为线上订单
            if order_type == EnumVal.ONLINE:
                is_online_ord_fulfill = True

            # 逐sku扣减库存
            for _, sku_obj in order_detail_df.iterrows():
                sku, quantity, sale_prc, cost_prc = sku_obj.sku_id, sku_obj.quantity, sku_obj.sale_price, sku_obj.cost_price

                # 记录扣减前库存
                ori_stock, ori_front_stock, ori_back_stock = solution.get_stocks(sku, store, dt)

                # 先扣减全场库存
                solution.sub(index=(sku, store, EnumVal.ALL, dt), delta_x=quantity)

                back_pick_cnt, front_pick_cnt, back_to_front_carry_cnt = 0, 0, 0
                # 线上订单：只有拣货成本，注意判断是否后场完单
                if order_type == EnumVal.ONLINE:
                    back_pick_cnt, front_pick_cnt = Evaluator.playback_online_sku(solution, sku, store, dt,
                                                                                  quantity)
                    ord_back_pick_cnt += back_pick_cnt
                    ord_front_pick_cnt += front_pick_cnt
                    if back_pick_cnt > 0:
                        ord_back_pick_kind += 1
                    if front_pick_cnt > 0:
                        ord_front_pick_kind += 1
                # 线下订单：只有搬运成本
                else:
                    back_to_front_carry_cnt = Evaluator.playback_offline_sku(solution, sku, store, dt, quantity)
                    ord_back_to_front_carry_cnt += back_to_front_carry_cnt
                    if back_to_front_carry_cnt > 0:
                        ord_back_to_front_carry_kind += 1

                # 记录扣减后库存
                cur_stock, cur_front_stock, cur_back_stock = solution.get_stocks(sku, store, dt)
                # log
                logger.info('[回放细节-商品] [商品{},销量{}][回放前库存:总{},前{},后{}][回放后库存:总{},前{},后{}][拣货数量:前{},后{}][搬运数量:{}] '. \
                            format(sku, quantity,
                                   round(ori_stock, Param.kpi_ndigits), round(ori_front_stock, Param.kpi_ndigits),
                                   round(ori_back_stock, Param.kpi_ndigits),
                                   round(cur_stock, Param.kpi_ndigits), round(cur_front_stock, Param.kpi_ndigits),
                                   round(cur_back_stock, Param.kpi_ndigits),
                                   front_pick_cnt, back_pick_cnt, back_to_front_carry_cnt))

                # dbg
                # if front_pick_cnt == 0 and back_pick_cnt == 0:
                #     print(ord_back_pick_kind, ord_front_pick_kind)
                #     raise(1)
                # if front_pick_cnt == 0 and back_pick_cnt > 0:
                #     print(ord_back_pick_kind, ord_front_pick_kind)
                #     raise(1)
                # if front_pick_cnt > 0 and back_pick_cnt == 0:
                #     print(ord_back_pick_kind, ord_front_pick_kind)
                #     raise(1)
                # if front_pick_cnt > 0 and back_pick_cnt > 0:
                #     print(ord_back_pick_kind, ord_front_pick_kind)
                #     raise(1)
                # if back_to_front_carry_cnt > 5 and back_to_front_carry_cnt < 10:
                #     print(ord_back_pick_kind, ord_front_pick_kind)
                #     raise(1)

            # 判定后场完单
            if order_type == EnumVal.ONLINE and ord_front_pick_kind == 0:
                is_ord_only_back_fulfill = True

        # 可履约订单
        if is_ord_fulfill:
            # 销售利润
            sale_profit = Helper.get_ord_sale_profit(order_detail_df)
            # 搬运成本
            carry_cost = Helper.get_ord_carry_cost(ord_back_to_front_carry_kind)
            # 拣货成本
            pickup_cost = Helper.get_ord_pickup_cost(ord_front_pick_kind, ord_back_pick_kind)
            # 履约成本
            fulfill_cost = carry_cost + pickup_cost
        # 不可履约订单
        else:
            # 缺货成本
            stockout_cost = Helper.get_ord_stockout_cost(order_detail_df)

        return is_ord_fulfill, is_online_ord_fulfill, is_ord_only_back_fulfill, sale_profit, stockout_cost, carry_cost, pickup_cost, fulfill_cost, ord_back_pick_kind, ord_front_pick_kind, ord_back_to_front_carry_kind, ord_back_pick_cnt, ord_front_pick_cnt, ord_back_to_front_carry_cnt

    @staticmethod
    def playback_online_sku(solution: Solution, sku: int, store: int, dt: str, quantity: float) -> (float, float):
        """,
        回放线上订单sku
        Args:
            solution: 解对象
            sku: 商品i
            store: 门店j
            dt: 日期t
            quantity: 销量

        Returns: 后场拣货种类，前场拣货种类

        """
        cur_back_stock = solution.get(index=(sku, store, EnumVal.BACK, dt))  # 当前后场库存
        # 定义拣货指标
        back_pick_cnt, front_pick_cnt = 0, 0
        if cur_back_stock >= quantity:  # 后场库存充足，则直接扣减后场库存
            solution.sub(index=(sku, store, EnumVal.BACK, dt), delta_x=quantity)
            back_pick_cnt += quantity
        elif cur_back_stock <= 0:  # 后场无库存，则直接扣减前场库存
            solution.sub(index=(sku, store, EnumVal.FRONT, dt), delta_x=quantity)
            front_pick_cnt += quantity
        else:  # 前后场组合拣货。优先扣减后场库存，再扣减前场库存
            solution.sub(index=(sku, store, EnumVal.BACK, dt), delta_x=cur_back_stock)
            solution.sub(index=(sku, store, EnumVal.FRONT, dt), delta_x=quantity - cur_back_stock)
            back_pick_cnt += cur_back_stock
            front_pick_cnt += (quantity - cur_back_stock)

        return back_pick_cnt, front_pick_cnt

    @staticmethod
    def playback_offline_sku(solution: Solution, sku: int, store: int, dt: str, quantity: float) -> float:
        """
        回放线下订单sku
        Args:
            solution: 解对象
            sku: 商品i
            store: 门店j
            dt: 日期t
            quantity: 销量

        Returns:

        """
        cur_front_stock = solution.get(index=(sku, store, EnumVal.FRONT, dt))  # 当前前场库存
        cur_back_stock = solution.get(index=(sku, store, EnumVal.BACK, dt))  # 当前后场库存
        # 定义搬运指标
        carry_cnt = 0
        # 如果前场库存库存充足，则直接扣减前场库存,不能取等号
        if cur_front_stock > quantity:
            solution.sub(index=(sku, store, EnumVal.FRONT, dt), delta_x=quantity)
        else:  # 先后场向前场搬货，再前场拣货
            carry_cnt = Param.get_carry_cnt(cur_back_stock, cur_front_stock, quantity)
            solution.sub(index=(sku, store, EnumVal.BACK, dt), delta_x=carry_cnt)
            solution.add(index=(sku, store, EnumVal.FRONT, dt), delta_x=carry_cnt)
            solution.sub(index=(sku, store, EnumVal.FRONT, dt), delta_x=quantity)
        return carry_cnt

    @staticmethod
    def judge(standardResultFile, userCommitFile, evalStrategy, recordId, n_jobs=4, to_csv=False):
        """
        评测入口
        Args:
            standardResultFile: 真实结果集(本题只)
            userCommitFile: 参赛者提交结果集
            evalStrategy: 0/1 A/B榜
            recordId: 举办方字段
            n_jobs: 并行数量
            to_csv: 是否将kpi输出至csv

        Returns: json

        """

        err_code, err_type, err_msg = -1, -1, ''
        score, score1, score2, score3 = Param.neg_inf, Param.neg_inf, Param.neg_inf, ''
        data_folder_path = standardResultFile  # 订单等数据所在文件夹，如: ./
        result_path = userCommitFile  # 选手提交文件
        ab = Param.A

        # A榜：0，B榜：1
        if evalStrategy == 1:
            ab = Param.B

        order_path = data_folder_path + 'test_sku_sales.csv'
        price_path = data_folder_path + 'test_sku_prices.csv'

        if get_filename(result_path) == False:
            err_code, err_type = 1, 1  # err_type = 1表示字符编码错误或者BOM问题,或者文件格式错误
            err_msg = "提交结果文件不是csv文件"
        elif is_empty_csv(result_path) == True:
            err_code, err_type = 2, 2  # err_type = 10表示文件存在重复行
            err_msg = "提交结果文件是空csv文件"
        else:
            is_success, err_msg, S, D, A, kpi_str = \
                Evaluator.run(order_path=order_path, result_path=result_path, price_path=price_path, ab=ab,
                              n_jobs=n_jobs, to_csv=to_csv)
            score, score1, score2, score3 = S, D, A, kpi_str
            if is_success:
                err_code, err_type = 0, 0  # 计算成功
            else:
                err_code, err_type = 4, 4  # 内部绘制校验不通过
                err_msg = err_msg + ' 详细指标为：' + score3

        result = {
            "recordId": recordId,
            "err_code": err_code,
            "err_type": err_type,
            "err_message": err_msg,
            "message": "",
            "score": score,
            "score_1": score1,
            "score_2": score2,
            "score_3": score3,
            "score_4": 0,
            "score_5": 0,
        }
        return result


### CCF大赛组提供评测程序
def get_filename(path):
    """
    判断提交格式是否为CSV
    :param path: 提交文件的路径
    :return:
    """
    if path.endswith(".csv"):
        return True
    else:
        return False


def is_empty_csv(path):
    """
    判断CSV文件是否为空
    :param path:
    :return:
    """
    import csv

    with open(path, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for i, _ in enumerate(reader):
            if i:  # found the second row
                return False
    return True


# 计算状态：-1待计算，0计算完成，1编码错误，2文件异常，3文件大小错误，4逻辑错误，5数据过时，6计算超时，7评测程序错误，8文件不完整,9字段错误，10表示文件存在重复行
# 11表示文件行数不一致，12表示提交结果超出最大或最小值
def judge(standardResultFile, userCommitFile, evalStrategy, recordId):
    """
    评测入口
    Args:
        standardResultFile: 真实结果集(本题只)
        userCommitFile: 参赛者提交结果集
        evalStrategy: 0/1 A/B榜
        recordId:

    Returns:

    """

    try:
        result = Evaluator.judge(standardResultFile, userCommitFile, evalStrategy, recordId)
    except:
        result = {
            "recordId": recordId,
            "err_code": 7,
            "err_type": 7,
            "err_message": '评测程序异常，请反馈~',
            "message": "",
            "score": 0,
            "score_1": 0,
            "score_2": 0,
            "score_3": 0,
            "score_4": 0,
            "score_5": 0,
        }
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":

    """
    评测文件夹结构
        -CCF_Eval
            test_sku_sales.csv  # 销量数据
            test_sku_prices.csv  # 价格数据，注意：评测数据中会新增ab_type列，取值为'A'或'B'，标注AB榜
            test_result.csv  # 参赛选手提供的答案
            judge.py  # 举办方线上评测代码
    """

    args = sys.argv[1:]
    if len(args) != 4:
        print("argument has error," + str(len(args)) + " not equal 4")
        print(args)
    else:
        result = judge(args[0], args[1], int(args[2]), int(args[3]))
        print(result)


