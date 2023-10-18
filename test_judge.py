# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName     : test_judye.py
# @Description  : test_judye实现，参赛选手自测入口
# @Time         : 2023/9/21 5:04 下午
# @Author       : JD.com
# @Software     : PyCharm

from judge import *


def check_ord_prc_df(order_df, price_df):
    """
    检查订单和价格数据
    Args:
        order_df: 订单
        price_df: 价格

    Returns:

    """

    is_legal = True
    # 行检查
    store_cnt = len(set(price_df['store_id'].values.tolist()))
    date_cnt = len(set(price_df['date'].values.tolist()))
    sku_cnt = len(set(price_df['sku_id'].values.tolist()))
    if len(price_df) != store_cnt * date_cnt * sku_cnt:
        print('价格数据行数错误，错误行数:{}, 要求行数:{}！'.format(len(price_df), 168000))
        is_legal = False

    # 列检查
    prc_cols = ['date', 'store_id', 'sku_id', 'salable_status', 'sale_price', 'cost_price', 'ab_type']
    if len(price_df.columns) != len(prc_cols):
        print('价格数据列数错误，错误列数:{}, 要求列数:{}！'.format(len(price_df.columns), len(prc_cols)))
        is_legal = False

    for col in price_df.columns:
        if col not in prc_cols:
            print('价格数据列名错误，错误列名:{}, 要求列名:{}！'.format(col, prc_cols))
            is_legal = False

    ord_cols = ['order_time', 'order_id', 'store_id', 'sku_id', 'quantity', 'channel']
    if len(order_df.columns) != len(ord_cols):
        print('订单数据列数错误，错误列数:{}, 要求列数:{}！'.format(len(price_df.columns), len(prc_cols)))
        is_legal = False

    for col in order_df.columns:
        if col not in ord_cols:
            print('订单数据列名错误，错误列名:{}, 要求列名:{}！'.format(col, prc_cols))
            is_legal = False

    # 空值检查
    if order_df.isnull().values.any():
        print('订单数据存在null')
        is_legal = False

    if price_df.isnull().values.any():
        print('价格数据存在null')
        is_legal = False


    # 正值检查
    check_df = price_df[(price_df['sale_price'] <= 0) | (price_df['cost_price'] <= 0)]
    if len(check_df) > 0:
        print('价格数据中的sale_price或者cost_price存在<=0的值')
        is_legal = False

    check_df = order_df[(order_df['quantity'] <= 0)]
    if len(check_df) > 0:
        print('订单数据quantity中存在<=0的值')
        is_legal = False


    # 主键重复检查
    order_df['date'] = order_df.apply(func=lambda x: x['order_time'][:10], axis=1)
    check_df = order_df[['date', 'store_id', 'sku_id', 'order_id']]
    duplicate_cnt = check_df.duplicated().sum()
    if duplicate_cnt > 0:
        print('订单数据存在重复主键，请检查:(date, store_id, sku_id, order_id)')
        is_legal = False

    check_df = price_df[['date', 'store_id', 'sku_id']]
    duplicate_cnt = check_df.duplicated().sum()
    if duplicate_cnt > 0:
        print('价格数据存在重复主键，请检查:(date, store_id, sku_id)')
        is_legal = False

    # 其他关键指标检查
    ab_set = set(price_df['ab_type'].values.tolist())
    if len(ab_set) != 2:
        print('价格数据ab_type枚举数量不等于2')
        is_legal = False
    if ('A' not in ab_set):
        print('价格数据ab_type枚举值没有A')
        is_legal = False
    if ('B' not in ab_set):
        print('价格数据ab_type枚举值没有B')
        is_legal = False

    check_df = price_df[(price_df['salable_status'] != 0) & (price_df['salable_status'] != 1)]
    if len(check_df) > 1:
        print('价格数据中salable_status枚举值存在除(0,1)之外的值')
        is_legal = False

    channel_set = set(order_df['channel'].values.tolist())
    if len(channel_set) != 2:
        print('订单数据中channel枚举数量不等于2')
        is_legal = False
    if (1 not in channel_set):
        print('订单数据中channel枚举值没有1')
        is_legal = False
    if (2 not in channel_set):
        print('订单数据中channel枚举值没有2')
        is_legal = False

    return is_legal

if __name__ == "__main__":

    """
    TO参赛选手文件夹结构
        -CCF_Eval
            test_sku_sales.csv  # 销量数据
            test_sku_prices.csv  # 价格数据，注意：评测数据中会新增ab_type列，取值为'A'或'B'，标注AB榜
            test_result.csv  # 参赛选手提供的答案
            judge.py  # 举办方线上评测代码
            test_judge.py  # 参赛选手自测入口
    """

    # 检查订单 价格数据
    order_df = pd.read_csv('test_sku_sales.csv', sep=',', encoding='utf8')
    price_df = pd.read_csv('test_sku_prices.csv', sep=',', encoding='utf8')
    if not check_ord_prc_df(order_df, price_df):
        raise('订单、价格数据检查未通过')

    # 评测数据中价格数据会新增ab_type列
    if 'ab_type' not in price_df.columns:
        price_df['ab_type'] = 'A'
        price_df.to_csv('sku_prices.csv', sep=',', encoding='utf8', index=False)

    # -------------------------------------------------------------
    standardResultFile = './'
    userCommitFile = './test_result.csv'
    evalStrategy = 0  # A 0；B 1
    recordId = 0

    # 并行评测
    logger.setLevel(logging.ERROR)
    result = Evaluator.judge(standardResultFile, userCommitFile, evalStrategy, recordId, n_jobs=8, to_csv=True)

    ## 串行评测，开log。参赛选手若想打印log，需设置logger.setLevel(logging.INFO)
    # logger.setLevel(logging.INFO)
    # result = Evaluator.judge(standardResultFile, userCommitFile, evalStrategy, recordId, n_jobs=1, to_csv=True)

    print(result)