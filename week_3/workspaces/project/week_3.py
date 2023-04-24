from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    String,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    RetryPolicy,
    RunRequest,
    ScheduleDefinition,
    SkipReason,
    graph,
    op,
    schedule,
    sensor,
    static_partitioned_config,
)
from workspaces.config import REDIS, S3
from workspaces.project.sensors import get_s3_keys
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    out={"stocks": Out(dagster_type=List[Stock])},
    required_resource_keys={"s3"},
    tags={"kind": "s3"},
)
def get_s3_data(context: OpExecutionContext):
    s3_key = context.op_config["s3_key"]
    stocks = [Stock.from_list(row) for row in context.resources.s3.get_data(s3_key)]
    return stocks


@op(
    ins={"stocks": In(List[Stock], description="List of Stock objects")},
    out={"aggregation": (Out(Aggregation, description="Single Aggregation object"))},
    description="Process Stocks",
)
def process_data(context: OpExecutionContext, stocks):
    max_stock = max(stocks, key=lambda stock: stock.high)
    res = Aggregation(date=max_stock.date, high=max_stock.high)
    context.log.info(res)
    return res


@op(
    ins={"aggregation": In(Aggregation, description="Single Aggregation object")},
    out=Out(Nothing),
    required_resource_keys={"redis"},
    tags={"kind": "redis"},
)
def put_redis_data(context: OpExecutionContext, aggregation):
    context.resources.redis.put_data(name=f"Agg:{aggregation.date}", value=aggregation)


@op(
    ins={"aggregation": In(Aggregation, description="Single Aggregation object")},
    out=Out(Nothing),
    required_resource_keys={"s3"},
    tags={"kind": "s3"},
)
def put_s3_data(context: OpExecutionContext, aggregation):
    context.resources.s3.put_data(f"Agg:{aggregation.date}", data=aggregation)


@graph
def machine_learning_graph():
    aggregation = process_data(get_s3_data())
    put_redis_data(aggregation)
    put_s3_data(aggregation)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}


docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}

MONTHS = list(map(str, list(range(1,11))))

@static_partitioned_config(partition_keys=MONTHS)
def docker_config(partition_key):
    return {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": f"prefix/stock_{partition_key}.csv"}}},
}


machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()},
    op_retry_policy=RetryPolicy(max_retries=10, delay=1),
)


machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker,
    resource_defs={"s3": s3_resource, "redis": redis_resource},
    op_retry_policy=RetryPolicy(max_retries=10, delay=1),
)


machine_learning_schedule_local = ScheduleDefinition(job=machine_learning_job_local, cron_schedule="*/15 * * * *")


@schedule(job=machine_learning_job_docker, cron_schedule="0 * * * *")
def machine_learning_schedule_docker():
    for key in MONTHS:
        yield RunRequest(run_key=f"month_{key}", partition_key=key)


@sensor(job=machine_learning_job_docker)
def machine_learning_sensor_docker():
    keys = get_s3_keys(bucket="dagster", prefix="prefix", endpoint_url="http://localstack:4566")

    if keys:
        for key_name in keys:
            yield RunRequest(
                run_key=key_name,
                run_config={
                    "resources": {
                        "s3": {"config": S3},
                        "redis": {"config": REDIS},
                    },
                    "ops": {"get_s3_data": {"config": {"s3_key": key_name}}},
                },
            )
    else:
        yield SkipReason("No new s3 files found in bucket.")
