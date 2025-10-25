import json
import os
from typing import Any

import duckdb
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from pydantic import BaseModel, Field

from schema_info import get_schema_info as load_schema_info
from utils import get_image_url
from allergen_detector import get_allergen_details

DB_PATH = "db/open_food.duckdb"
_DEFAULT_USER_ID = "1649954c-dca1-456d-a28f-4d4527c997d8"

_food_query_agent: CompiledStateGraph | None = None


class FoodResponse(BaseModel):

    codes: list[str] = Field(description="A list of unique codes identifying the food products.")


class SQLResponse(BaseModel):

    reasoning: str = Field(description="The step-by-step reasoning and thought process that leads to the answer below.")
    answer: str = Field(description="The final SQL SELECT query that answers the user's question.")


def _generate_sql_tool(prompt: str) -> str:
    """
    Generates a DuckDB SQL SELECT query based on the user's question.

    Args:
        prompt (str): The natural language question to convert into a SQL query.

    Returns:
        str: The generated SQL query as a string.
    """

    schema_summary = load_schema_info(DB_PATH)

    system_prompt = (
        "You are an expert SQL query generator. Your task is to convert natural "
        "language questions into DuckDB SQL SELECT queries using the provided "
        "database schema. Ensure the queries include random sampling by using "
        "`ORDER BY random()`. Think step-by-step before providing the final SQL "
        "query."
    )

    chat_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Schema: {schema_info}"),
        ("human", "Instruction: Limit the retrieval to max 10 records."),
        ("human", "Question: {prompt}"),
    ])

    chat_prompt = chat_template.invoke({
        "schema_info": schema_summary,
        "prompt": prompt,
    })

    print(chat_prompt.to_string())

    model = (
        init_chat_model("gpt-4.1-2025-04-14", model_provider="openai")
        .with_structured_output(SQLResponse)
    )

    sql_response: SQLResponse = model.invoke(chat_prompt)

    print(f"Reasoning: {sql_response.reasoning}")
    print(f"SQL: {sql_response.answer}")

    return sql_response.answer


def _execute_sql_tool(query: str) -> str:
    """
    Executes a SQL query on a DuckDB database and returns the results as a JSON
    string.

    Args:
        query (str): The SQL query to be executed.

    Returns:
        str: A JSON string representing the first 100 rows of the query result,
             formatted as a list of records (each record is a dictionary).
    """

    with duckdb.connect(DB_PATH, read_only=True) as conn:
        return conn.query(query).df().head(100).to_json(orient="records")


def _get_food_query_agent() -> CompiledStateGraph:
    global _food_query_agent

    if _food_query_agent is None:
        _food_query_agent = create_react_agent(
            model="gpt-4.1-2025-04-14",
            tools=[_generate_sql_tool, _execute_sql_tool],
            prompt="Generate SQL and execute is to retrieve food name and code.",
            response_format=FoodResponse,
        )

    return _food_query_agent


@task
def find_food_products(codes: list[str]) -> list[dict[str, Any]]:

    if not codes:
        return []

    placeholders = ", ".join("?" for _ in codes)
    query = f"SELECT * FROM products WHERE code IN ({placeholders})"

    with duckdb.connect(DB_PATH, read_only=True) as conn:
        return conn.sql(query, params=codes).df().to_dict(orient="records")


@task
def query_food(prompt: str) -> list[dict[str, Any]]:

    agent = _get_food_query_agent()

    result = agent.invoke({"messages": HumanMessage(content=prompt)})

    print(f"result: {result}")

    food_response: FoodResponse = result["structured_response"]

    print(f"food_response: {food_response}")

    food_products = find_food_products(food_response.codes).result()

    user_id = os.getenv("NUTRIFOODBOT_USER_ID", _DEFAULT_USER_ID)
    try:
        if not os.getenv("NUTRIFOODBOT_USER_ID"):
            import streamlit as st  # type: ignore
            session_user = st.session_state.get("user_id")
            if session_user:
                user_id = str(session_user)
    except Exception:
        pass

    if food_products:
        allergen_cache: dict[str, tuple[str, ...]] = {}
        enriched_products: list[dict[str, Any]] = []
        for product in food_products:
            code = str(product.get("code") or "")
            allergens: tuple[str, ...] = ()
            if code:
                cached = allergen_cache.get(code)
                if cached is None:
                    details = get_allergen_details(user_id, code)
                    detected = details.get("detected_allergens") or ()
                    cached = tuple(str(a) for a in detected)
                    allergen_cache[code] = cached
                allergens = cached
            enriched_products.append({**product, "allergens": list(allergens)})
        food_products = enriched_products

    print(f"food_products: {food_products}")

    return food_products
