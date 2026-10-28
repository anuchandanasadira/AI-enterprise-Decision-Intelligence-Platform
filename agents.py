import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = None

if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
        print("Gemini connected successfully")
    except Exception as e:
        print("Gemini connection failed:", e)


# =========================================================
# LOAD DATASETS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")


def load_csv(filename):

    path = os.path.join(DATASET_DIR, filename)

    try:
        df = pd.read_csv(path)

        print(f"{filename}: {len(df)} rows loaded")

        return df

    except Exception as e:

        print(f"Error loading {filename}:", e)

        return pd.DataFrame()


sales_df = load_csv("sales.csv")
inventory_df = load_csv("inventory.csv")
reviews_df = load_csv("customer_reviews.csv")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_column(df, possible_names):

    if df.empty:
        return None

    # Exact match
    for name in possible_names:

        for column in df.columns:

            if column.lower().strip() == name.lower().strip():

                return column

    # Partial match
    for name in possible_names:

        for column in df.columns:

            if name.lower() in column.lower():

                return column

    return None


def clean_number(value):

    try:
        return float(value)

    except:

        return 0


# =========================================================
# SALES AGENT
# =========================================================

def sales_agent(question):

    if sales_df.empty:

        return {
            "status": "No sales data available",
            "records": 0,
            "average_quantity": 0,
            "maximum_quantity": 0,
            "total_sales": None
        }


    quantity_col = find_column(
        sales_df,
        [
            "Quantity",
            "quantity",
            "Qty",
            "Units Sold"
        ]
    )


    sales_col = find_column(
        sales_df,
        [
            "Sales",
            "Sales Amount",
            "Revenue",
            "Total Sales",
            "Amount"
        ]
    )


    result = {

        "status": "Sales analysis completed",

        "records": len(sales_df),

        "average_quantity": 0,

        "maximum_quantity": 0,

        "total_sales": None
    }


    # Quantity analysis

    if quantity_col:

        quantity_values = pd.to_numeric(
            sales_df[quantity_col],
            errors="coerce"
        ).dropna()


        if len(quantity_values) > 0:

            result["average_quantity"] = round(
                quantity_values.mean(),
                2
            )


            result["maximum_quantity"] = int(
                quantity_values.max()
            )


    # Sales analysis

    if sales_col:

        sales_values = pd.to_numeric(
            sales_df[sales_col],
            errors="coerce"
        ).dropna()


        if len(sales_values) > 0:

            result["total_sales"] = round(
                sales_values.sum(),
                2
            )


    return result


# =========================================================
# INVENTORY AGENT
# =========================================================

def inventory_agent(question):

    if inventory_df.empty:

        return {
            "status": "No inventory data available",
            "total_products": 0,
            "low_stock_count": 0,
            "products": []
        }


    # -----------------------------------------------------
    # FIND STOCK COLUMN
    # -----------------------------------------------------

    stock_col = find_column(
        inventory_df,
        [
            "Stock Quantity",
            "stock_quantity",
            "Stock",
            "Quantity",
            "Inventory"
        ]
    )


    # -----------------------------------------------------
    # FIND PRODUCT NAME COLUMN
    # -----------------------------------------------------

    product_col = find_column(
        inventory_df,
        [
            "Product Name",
            "Product",
            "product_name",
            "ProductName",
            "Item Name",
            "item_name",
            "Item",
            "Name",
            "Product Description",
            "product"
        ]
    )


    # -----------------------------------------------------
    # AUTOMATIC FALLBACK FOR PRODUCT COLUMN
    # -----------------------------------------------------

    if not product_col:

        excluded_columns = {

            stock_col,

            "Product ID",
            "ProductID",
            "product_id",

            "ID",
            "id",

            "SKU",
            "sku"
        }


        for column in inventory_df.columns:

            if column in excluded_columns:

                continue


            # Look for text columns

            if inventory_df[column].dtype == "object":

                sample_values = (
                    inventory_df[column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )


                if len(sample_values) > 0:

                    column_lower = column.lower()


                    if (
                        "id" not in column_lower
                        and
                        "sku" not in column_lower
                    ):

                        product_col = column

                        break


    # -----------------------------------------------------
    # STOCK COLUMN CHECK
    # -----------------------------------------------------

    if not stock_col:

        return {

            "status": "Stock column not found",

            "total_products": len(inventory_df),

            "low_stock_count": 0,

            "products": []
        }


    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    data = inventory_df.copy()


    data["_stock"] = pd.to_numeric(
        data[stock_col],
        errors="coerce"
    )


    data = data.dropna(
        subset=["_stock"]
    )


    # Lowest stock first

    data = data.sort_values(
        "_stock"
    )


    products = []


    # -----------------------------------------------------
    # ANALYZE PRODUCTS
    # -----------------------------------------------------

    for _, row in data.iterrows():

        stock = float(
            row["_stock"]
        )


        # PRODUCT NAME

        if product_col:

            product_name = str(
                row[product_col]
            ).strip()

        else:

            product_name = "Unknown Product"


        # Avoid blank names

        if (
            not product_name
            or
            product_name.lower() == "nan"
        ):

            product_name = "Unknown Product"


        # -------------------------------------------------
        # RISK CLASSIFICATION
        # -------------------------------------------------

        if stock <= 15:

            risk = "HIGH"

            reason = (
                "Very low stock. "
                "Product may run out soon."
            )

            action = (
                "Restock immediately."
            )


        elif stock <= 30:

            risk = "MEDIUM"

            reason = (
                "Stock level is getting low."
            )

            action = (
                "Plan restocking and monitor stock."
            )


        else:

            risk = "LOW"

            reason = (
                "Stock level is currently sufficient."
            )

            action = (
                "Continue regular monitoring."
            )


        # -------------------------------------------------
        # STORE PRODUCT INFORMATION
        # -------------------------------------------------

        products.append({

            "product": product_name,

            "stock": int(stock),

            "risk": risk,

            "reason": reason,

            "action": action

        })


    # -----------------------------------------------------
    # LOW STOCK PRODUCTS
    # -----------------------------------------------------

    low_stock_products = [

        p for p in products

        if p["stock"] <= 30

    ]


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {

        "status":
            "Inventory analysis completed",

        "total_products":
            len(products),

        "low_stock_count":
            len(low_stock_products),

        "products":
            products[:10]

    }


# =========================================================
# REVIEW AGENT
# =========================================================

def review_agent(question):

    if reviews_df.empty:

        return {

            "status":
                "No customer review data available",

            "total_reviews": 0,

            "negative_reviews": 0,

            "other_reviews": 0,

            "negative_percentage": 0,

            "themes": []

        }


    total_reviews = len(
        reviews_df
    )


    sentiment_col = find_column(

        reviews_df,

        [
            "Sentiment",
            "sentiment"
        ]

    )


    text_col = find_column(

        reviews_df,

        [
            "Text",
            "Review",
            "Review Text",
            "Comment",
            "Customer Review"
        ]

    )


    negative_reviews = 0


    # -----------------------------------------------------
    # SENTIMENT COLUMN
    # -----------------------------------------------------

    if sentiment_col:

        sentiments = (

            reviews_df[
                sentiment_col
            ]

            .astype(str)

            .str.lower()

            .str.strip()

        )


        negative_reviews = (

            sentiments
            .str.contains(
                "negative|neg"
            )
            .sum()

        )


    # -----------------------------------------------------
    # REVIEW TEXT
    # -----------------------------------------------------

    elif text_col:

        negative_words = [

            "bad",
            "poor",
            "worst",
            "slow",
            "delay",
            "late",
            "problem",
            "issue",
            "disappointed",
            "terrible",
            "hate",
            "broken",
            "damage",
            "not good"

        ]


        for text in reviews_df[
            text_col
        ].astype(str):

            text_lower = text.lower()


            if any(
                word in text_lower
                for word in negative_words
            ):

                negative_reviews += 1


    other_reviews = (
        total_reviews
        -
        negative_reviews
    )


    negative_percentage = 0


    if total_reviews > 0:

        negative_percentage = round(

            (
                negative_reviews
                /
                total_reviews
            )
            *
            100,

            1

        )


    # =====================================================
    # CUSTOMER COMPLAINT THEMES
    # =====================================================

    themes = []


    if text_col:

        theme_keywords = {

            "Delivery": [

                "delivery",
                "shipping",
                "late",
                "delay"

            ],

            "Product Quality": [

                "quality",
                "broken",
                "damage",
                "defect"

            ],

            "Customer Service": [

                "support",
                "service",
                "customer care"

            ],

            "Price": [

                "price",
                "expensive",
                "cost"

            ],

            "Website / App": [

                "website",
                "app",
                "login",
                "page"

            ]

        }


        text_data = (

            reviews_df[text_col]

            .astype(str)

            .str.lower()

        )


        for theme, keywords in (
            theme_keywords.items()
        ):

            count = 0


            for text in text_data:

                if any(
                    word in text
                    for word in keywords
                ):

                    count += 1


            if count > 0:

                themes.append({

                    "theme": theme,

                    "count": count

                })


    return {

        "status":
            "Customer review analysis completed",

        "total_reviews":
            total_reviews,

        "negative_reviews":
            int(
                negative_reviews
            ),

        "other_reviews":
            int(
                other_reviews
            ),

        "negative_percentage":
            negative_percentage,

        "themes":
            themes

    }


# =========================================================
# MULTI-AGENT AI
# =========================================================

def run_multi_agent_analysis(question):

    print(
        "\nRunning Sales Agent..."
    )

    sales_result = sales_agent(
        question
    )


    print(
        "Running Inventory Agent..."
    )

    inventory_result = inventory_agent(
        question
    )


    print(
        "Running Review Agent..."
    )

    review_result = review_agent(
        question
    )


    result = {

        "sales":
            sales_result,

        "inventory":
            inventory_result,

        "reviews":
            review_result,

        "agents": {

            "sales_agent":
                "Completed",

            "inventory_agent":
                "Completed",

            "review_agent":
                "Completed",

            "decision_agent":
                "Ready"

        }

    }


    return result


# =========================================================
# DECISION AGENT
# =========================================================

def decision_agent(
    question,
    agent_results
):

    data_for_ai = json.dumps(
        agent_results,
        indent=2
    )


    prompt = f"""

You are the Decision Agent of an
AI Enterprise Decision Intelligence Platform.

Business question:

{question}


The Sales Agent, Inventory Agent and
Review Agent produced this data:

{data_for_ai}


Give a simple business answer.


Rules:

1. Use simple English.

2. Do not mention APIs, Gemini,
FAISS, RAG, Python or technical errors.

3. Give the most important finding first.

4. Mention actual product names and
numbers when available.

5. Give reasons for the finding.

6. Give practical business recommendations.

7. Keep the answer short and clear.

8. Use bullet points.

"""


    # =====================================================
    # GEMINI RESPONSE
    # =====================================================

    if client:

        try:

            response = client.models.generate_content(

                model="gemini-3-flash-preview",

                contents=prompt

            )


            if response and response.text:

                return response.text.strip()


        except Exception as e:

            print(
                "Decision Agent temporarily unavailable."
            )


    # =====================================================
    # SAFE FALLBACK
    # =====================================================

    inventory = agent_results.get(
        "inventory",
        {}
    )


    sales = agent_results.get(
        "sales",
        {}
    )


    reviews = agent_results.get(
        "reviews",
        {}
    )


    products = inventory.get(
        "products",
        []
    )


    high_risk = [

        p for p in products

        if p.get("risk") == "HIGH"

    ]


    medium_risk = [

        p for p in products

        if p.get("risk") == "MEDIUM"

    ]


    answer_parts = []


    # -----------------------------------------------------
    # MAIN FINDING
    # -----------------------------------------------------

    if high_risk:

        names = ", ".join(

            p["product"]

            for p in high_risk[:5]

        )


        answer_parts.append(

            f"Key Finding:\n"
            f"{len(high_risk)} product(s) "
            f"have high inventory risk: "
            f"{names}."

        )


        answer_parts.append(

            "Priority Action:\n"
            "Restock high-risk products immediately."

        )


    elif medium_risk:

        answer_parts.append(

            f"Key Finding:\n"
            f"{len(medium_risk)} product(s) "
            f"have medium inventory risk."

        )


        answer_parts.append(

            "Priority Action:\n"
            "Plan restocking and monitor inventory closely."

        )


    else:

        answer_parts.append(

            "Key Finding:\n"
            "Inventory levels are currently stable."

        )


    # -----------------------------------------------------
    # SALES
    # -----------------------------------------------------

    if sales.get("records"):

        answer_parts.append(

            f"Sales Insight:\n"
            f"{sales['records']} sales records "
            f"were analyzed. "
            f"Average quantity per record is "
            f"{sales.get('average_quantity', 0)}."

        )


    # -----------------------------------------------------
    # CUSTOMER REVIEWS
    # -----------------------------------------------------

    if reviews.get("total_reviews"):

        answer_parts.append(

            f"Customer Feedback:\n"
            f"{reviews['negative_reviews']} out of "
            f"{reviews['total_reviews']} reviews "
            f"are negative "
            f"({reviews['negative_percentage']}%)."

        )


    # -----------------------------------------------------
    # RECOMMENDATION
    # -----------------------------------------------------

    answer_parts.append(

        "Recommendation:\n"
        "Prioritize low-stock products, "
        "monitor sales demand, and review "
        "customer feedback regularly."

    )


    return "\n\n".join(
        answer_parts
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    question = (
        "Which products have low stock?"
    )


    print(
        "\n========================================"
    )

    print(
        "MULTI-AGENT AI TEST"
    )

    print(
        "========================================"
    )


    results = run_multi_agent_analysis(
        question
    )


    print(
        json.dumps(
            results,
            indent=2
        )
    )


    print(
        "\n========================================"
    )

    print(
        "DECISION AGENT"
    )

    print(
        "========================================"
    )


    answer = decision_agent(

        question,

        results

    )


    print(answer)