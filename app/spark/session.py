"""
SparkSession factory — creates or returns the cached session.
"""

import streamlit as st
from pyspark.sql import SparkSession

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SPARK_APP_NAME, SPARK_MASTER, SPARK_LOG_LEVEL


@st.cache_resource(show_spinner="🔧 Initialising Spark…")
def get_spark() -> SparkSession:
    """
    Create (or return cached) SparkSession.
    Configured for local mode with sensible defaults.
    """
    spark = (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.extraJavaOptions", "-Dderby.system.home=/tmp/derby")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(SPARK_LOG_LEVEL)
    return spark


def stop_spark():
    """Gracefully stop the Spark session."""
    try:
        spark = SparkSession.getActiveSession()
        if spark:
            spark.stop()
    except Exception:
        pass
