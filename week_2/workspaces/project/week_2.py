from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    String,
    graph,
    op,
)
from workspaces.config import REDIS, S3, S3_FILE
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    out={"stocks": Out(dagster_type=List[Stock])},
    required_resource_keys={"s3"},
    tags={"kind": "s3"},
)
def get_s3_data(context:OpExecutionContext):
    s3_key = context.op_config['s3_key']
    # stocks = list(context.resources.s3.get_data(s3_key))
    stocks = [Stock.from_list(record) for record in context.resources.s3.get_data(s3_key)]
    return stocks


@op(
    ins={"stocks": In(List[Stock], description="List of Stock objects")},
    out={"aggregation":(Out(Aggregation, description="Single Aggregation object"))},
    description="Process Stocks"
)
def process_data(context:OpExecutionContext, stocks):
    max_stock = max(stocks, key=lambda stock: stock.high)
    res = Aggregation(date=max_stock.date, high=max_stock.high)
    context.log.info(res)
    return res


@op(
    ins={"agg": In(Aggregation, description="Single Aggregation object")}, 
    out=Out(Nothing),
    required_resource_keys={"redis"},
    tags={"kind": "redis"},
)
def put_redis_data(context:OpExecutionContext, agg):
    context.resources.redis.put_data(name="Aggregation", value=agg)



@op(
    ins={"agg": In(Aggregation, description="Single Aggregation object")}, 
    out=Out(Nothing),
    required_resource_keys={"s3"},
    tags={"kind": "s3"},
)
def put_s3_data(context:OpExecutionContext, agg):
    context.resources.s3.put_data("Aggregation", data=agg)


@graph
def machine_learning_graph():
    agg = process_data(get_s3_data())
    put_redis_data(agg)
    put_s3_data(agg)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()}
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker,
    resource_defs={"s3": s3_resource, "redis": redis_resource}
)
