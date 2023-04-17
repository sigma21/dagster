import csv
from datetime import datetime
from typing import Iterator, List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    String,
    graph,
    job,
    op,
    usable_as_dagster_type,
)
from pydantic import BaseModel


@usable_as_dagster_type(description="Stock data")
class Stock(BaseModel):
    date: datetime
    close: float
    volume: int
    open: float
    high: float
    low: float

    @classmethod
    def from_list(cls, input_list: List[str]):
        """Do not worry about this class method for now"""
        return cls(
            date=datetime.strptime(input_list[0], "%Y/%m/%d"),
            close=float(input_list[1]),
            volume=int(float(input_list[2])),
            open=float(input_list[3]),
            high=float(input_list[4]),
            low=float(input_list[5]),
        )


@usable_as_dagster_type(description="Aggregation of stock data")
class Aggregation(BaseModel):
    date: datetime
    high: float


def csv_helper(file_name: str) -> Iterator[Stock]:
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield Stock.from_list(row)


@op(config_schema={"s3_key": String}, out=Out(List[Stock], description="List of Stock objects"), description="Extract Stocks data from storage layer")
def get_s3_data_op(context: OpExecutionContext) -> List[Stock]:
    stocks = list(csv_helper(context.op_config["s3_key"]))
    return stocks


@op(
    ins={"stocks": In(List[Stock], description="List of Stock objects")},
    out={"aggregation":(Out(Aggregation, description="Single Aggregation object"))},
    description="Process Stocks"
)
def process_data_op(context: OpExecutionContext, stocks: List[Stock]) -> Aggregation:
    max_stock = max(stocks, key=lambda stock: stock.high)
    res = Aggregation(date=max_stock.date, high=max_stock.high)
    context.log.info(res)
    return res


@op(ins={"agg": In(Aggregation, description="Single Aggregation object")}, out=Out(Nothing), description="Upload to caching layer")
def put_redis_data_op(agg: Aggregation) -> Nothing:
    pass


@op(ins={"agg": In(Aggregation, description="Single Aggregation object")}, out=Out(Nothing), description="Upload to storage layer")
def put_s3_data_op(agg: Aggregation) -> Nothing:
    pass


@graph
def machine_learning_graph():
    agg = process_data_op(get_s3_data_op())
    put_redis_data_op(agg)
    put_s3_data_op(agg)


machine_learning_job = machine_learning_graph.to_job(config={"ops": {"get_s3_data_op": {"config": {"s3_key": "week_1/data/stock.csv"}}}})


# or if the config will be provided through UI
#
# @job
# def machine_learning_job():
#     a = process_data_op(get_s3_data_op())
#     put_redis_data_op(a)
#     put_s3_data_op(a)
