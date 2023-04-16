import csv
from datetime import datetime
import heapq
from typing import Iterator, List

from dagster import (
    Any,
    DynamicOut,
    DynamicOutput,
    In,
    Nothing,
    OpExecutionContext,
    Out,
    Output,
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


@op(
    config_schema={"s3_key": String},
    out={"empty": Out(Nothing), "not_empty": Out(List[Stock], description="List of Stock objects")},
    description="Extract Stocks data from storage layer",
)
def get_s3_data_op(context) -> Nothing | List[Stock]:
    stocks = list(csv_helper(context.op_config["s3_key"]))

    if not stocks:
        yield Output(Nothing, "empty")
    else:
        yield Output(stocks, "not_empty")


@op(
    config_schema={"nlargest": int},
    ins={"stocks": In(List[Stock], description="List of Stock objects")},
    out=(DynamicOut(Aggregation, description="N amount of Aggregation object(s)")),
    description="Process Stocks dynamically",
)
def process_data_op(context, stocks: List[Stock]) -> Aggregation:
    nlargest = context.op_config["nlargest"]
    top_n_stocks = heapq.nlargest(nlargest, stocks, key=lambda agg: agg.high)

    for i, stock in enumerate(top_n_stocks):
        agg = Aggregation(date=stock.date, high=stock.high)
        context.log.info(f"Top agg #{i+1}: {agg}")
        yield DynamicOutput(agg, mapping_key=f"Agg_{i+1}")


@op(
    ins={"agg": In(Aggregation, description="Single Aggregation object")},
    out=Out(Nothing),
    description="Upload to caching layer",
)
def put_redis_data_op(agg: Aggregation) -> Nothing:
    pass


@op(
    ins={"agg": In(Aggregation, description="Single Aggregation object")},
    out=Out(Nothing),
    description="Upload to storage layer",
)
def put_s3_data_op(agg: Aggregation) -> Nothing:
    pass


@op(
    ins={"empty_stocks": In(dagster_type=Any)},
    out=Out(Nothing),
    description="Notify if stock list is empty",
)
def empty_stock_notify_op(context: OpExecutionContext, empty_stocks: Any) -> Nothing:
    context.log.info("No stocks returned")


@graph
def machine_learning_dynamic_job():
    empty, not_empty = get_s3_data_op()
    empty_stock_notify_op(empty)

    aggs = process_data_op(not_empty)
    aggs.map(put_redis_data_op)
    aggs.map(put_s3_data_op)


job = machine_learning_dynamic_job.to_job(
    config={
        "ops": {
            "get_s3_data_op": {"config": {"s3_key": "week_1/data/stock.csv"}},
            "process_data_op": {"config": {"nlargest": 3}},
        }
    }
)
# job = machine_learning_dynamic_job.to_job(config={"ops": {"get_s3_data_op": {"config": {"s3_key": "week_1/data/empty_stock.csv"}}}})


# @job
# def machine_learning_dynamic_job():
#     empty, not_empty = get_s3_data_op()
#     empty_stock_notify_op(empty)

#     aggs = process_data_op(not_empty)
#     aggs.map(put_redis_data_op)
#     aggs.map(put_s3_data_op)
