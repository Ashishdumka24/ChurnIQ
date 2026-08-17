# ============================================================
# ChurnIQ - MySQL Database Module
# ============================================================

import os
from pathlib import Path

import mysql.connector
from mysql.connector import Error


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.getenv("CHURNIQ_DB_HOST", "localhost"),
    "port": int(os.getenv("CHURNIQ_DB_PORT", "3306")),
    "user": os.getenv("CHURNIQ_DB_USER", "root"),
    "password": os.getenv("CHURNIQ_DB_PASSWORD", "Aasnhjiaslhi@123"),
    "database": os.getenv("CHURNIQ_DB_NAME", "churniq"),
}


# ============================================================
# CONNECTION
# ============================================================

def get_connection():
    """
    Create and return a MySQL database connection.
    """

    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
        )

        if connection.is_connected():
            return connection

    except Error as exc:
        print(f"MySQL connection error: {exc}")

    return None


# ============================================================
# CONNECTION TEST
# ============================================================

def test_connection():
    """
    Test whether ChurnIQ can connect to MySQL.
    Returns True or False.
    """

    connection = None

    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
        )

        if connection.is_connected():
            return True

    except Error as exc:
        print(f"MySQL test failed: {exc}")
        return False

    finally:
        if connection is not None and connection.is_connected():
            connection.close()

    return False


# ============================================================
# CREATE CUSTOMER TABLE
# ============================================================

def create_customer_table():
    """
    Create the main customer table if it does not exist.
    """

    connection = get_connection()

    if connection is None:
        return False

    cursor = None

    try:
        cursor = connection.cursor()

        query = """
        CREATE TABLE IF NOT EXISTS customers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(100) UNIQUE,
            gender VARCHAR(50),
            senior_citizen INT,
            partner VARCHAR(50),
            dependents VARCHAR(50),
            tenure INT,
            phone_service VARCHAR(50),
            multiple_lines VARCHAR(100),
            internet_service VARCHAR(100),
            online_security VARCHAR(100),
            online_backup VARCHAR(100),
            device_protection VARCHAR(100),
            tech_support VARCHAR(100),
            streaming_tv VARCHAR(100),
            streaming_movies VARCHAR(100),
            contract VARCHAR(100),
            paperless_billing VARCHAR(50),
            payment_method VARCHAR(150),
            monthly_charges DECIMAL(12,2),
            total_charges DECIMAL(12,2),
            churn VARCHAR(50),
            churn_probability DECIMAL(8,2),
            prediction VARCHAR(50),
            risk_level VARCHAR(100),
            risk_score DECIMAL(8,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        cursor.execute(query)
        connection.commit()

        return True

    except Error as exc:
        print(f"Could not create customer table: {exc}")
        return False

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


# ============================================================
# CREATE SYSTEM TABLE
# ============================================================

def create_system_tables():
    """
    Create all required ChurnIQ tables.
    """

    return create_customer_table()


# ============================================================
# INSERT CUSTOMER
# ============================================================

def insert_customer(customer):
    """
    Insert one customer dictionary into MySQL.
    """

    connection = get_connection()

    if connection is None:
        return False

    cursor = None

    try:
        cursor = connection.cursor()

        query = """
        INSERT INTO customers (
            customer_id,
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
            churn,
            churn_probability,
            prediction,
            risk_level,
            risk_score
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        """

        values = (
            customer.get("customer_id"),
            customer.get("gender"),
            customer.get("senior_citizen"),
            customer.get("partner"),
            customer.get("dependents"),
            customer.get("tenure"),
            customer.get("phone_service"),
            customer.get("multiple_lines"),
            customer.get("internet_service"),
            customer.get("online_security"),
            customer.get("online_backup"),
            customer.get("device_protection"),
            customer.get("tech_support"),
            customer.get("streaming_tv"),
            customer.get("streaming_movies"),
            customer.get("contract"),
            customer.get("paperless_billing"),
            customer.get("payment_method"),
            customer.get("monthly_charges"),
            customer.get("total_charges"),
            customer.get("churn"),
            customer.get("churn_probability"),
            customer.get("prediction"),
            customer.get("risk_level"),
            customer.get("risk_score"),
        )

        cursor.execute(query, values)
        connection.commit()

        return True

    except Error as exc:
        print(f"Customer insert failed: {exc}")
        return False

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


# ============================================================
# GET CUSTOMERS
# ============================================================

def get_customers(limit=1000):
    """
    Retrieve customers from MySQL.
    """

    connection = get_connection()

    if connection is None:
        return []

    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT *
        FROM customers
        ORDER BY id DESC
        LIMIT %s
        """

        cursor.execute(query, (int(limit),))

        return cursor.fetchall()

    except Error as exc:
        print(f"Could not retrieve customers: {exc}")
        return []

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


# ============================================================
# CUSTOMER COUNT
# ============================================================

def get_customer_count():
    """
    Return total number of customers.
    """

    connection = get_connection()

    if connection is None:
        return 0

    cursor = None

    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM customers"
        )

        result = cursor.fetchone()

        return int(result[0]) if result else 0

    except Error as exc:
        print(f"Could not count customers: {exc}")
        return 0

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


# ============================================================
# DELETE ALL CUSTOMERS
# ============================================================

def clear_customers():
    """
    Delete all customer records.
    """

    connection = get_connection()

    if connection is None:
        return False

    cursor = None

    try:
        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM customers"
        )

        connection.commit()

        return True

    except Error as exc:
        print(f"Could not clear customers: {exc}")
        return False

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():
    """
    Initialize ChurnIQ database tables.
    """

    try:
        return create_system_tables()

    except Exception as exc:
        print(
            f"Database initialization failed: {exc}"
        )
        return False


# ============================================================
# MAIN TEST
# ============================================================

def save_dataframe(df):
    """
    Save a pandas DataFrame into MySQL.
    Handles IBM Telco dataset and ChurnIQ prediction output.
    """

    if df is None or df.empty:
        print("No data to save.")
        return 0

    import pandas as pd

    connection = get_connection()

    if connection is None:
        print("MySQL connection failed.")
        return 0

    cursor = None

    try:
        # ----------------------------------------------------
        # Ensure table exists
        # ----------------------------------------------------

        create_customer_table()

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Normalize column names
        # ----------------------------------------------------

        normalized = {}

        for column in df.columns:

            key = (
                str(column)
                .strip()
                .lower()
                .replace(" ", "")
                .replace("_", "")
                .replace("-", "")
            )

            normalized[key] = column

        # ----------------------------------------------------
        # Helper
        # ----------------------------------------------------

        def value(row, *names):

            for name in names:

                key = (
                    str(name)
                    .strip()
                    .lower()
                    .replace(" ", "")
                    .replace("_", "")
                    .replace("-", "")
                )

                if key in normalized:

                    result = row[
                        normalized[key]
                    ]

                    if pd.isna(result):
                        return None

                    return result

            return None

        # ----------------------------------------------------
        # Customer ID
        # ----------------------------------------------------

        customer_column = None

        for possible in [
            "customerID",
            "CustomerID",
            "Customer ID",
            "customer_id",
        ]:

            key = (
                possible
                .strip()
                .lower()
                .replace(" ", "")
                .replace("_", "")
                .replace("-", "")
            )

            if key in normalized:

                customer_column = normalized[key]
                break

        if customer_column is None:

            print(
                "Customer ID column not found."
            )

            print(
                "Available columns:"
            )

            print(
                list(df.columns)
            )

            return 0

        # ----------------------------------------------------
        # SQL
        # ----------------------------------------------------

        query = """
        INSERT INTO customers (
            customer_id,
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
            churn,
            churn_probability,
            prediction,
            risk_level,
            risk_score
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            gender = VALUES(gender),
            senior_citizen = VALUES(senior_citizen),
            partner = VALUES(partner),
            dependents = VALUES(dependents),
            tenure = VALUES(tenure),
            phone_service = VALUES(phone_service),
            multiple_lines = VALUES(multiple_lines),
            internet_service = VALUES(internet_service),
            online_security = VALUES(online_security),
            online_backup = VALUES(online_backup),
            device_protection = VALUES(device_protection),
            tech_support = VALUES(tech_support),
            streaming_tv = VALUES(streaming_tv),
            streaming_movies = VALUES(streaming_movies),
            contract = VALUES(contract),
            paperless_billing = VALUES(paperless_billing),
            payment_method = VALUES(payment_method),
            monthly_charges = VALUES(monthly_charges),
            total_charges = VALUES(total_charges),
            churn = VALUES(churn),
            churn_probability = VALUES(churn_probability),
            prediction = VALUES(prediction),
            risk_level = VALUES(risk_level),
            risk_score = VALUES(risk_score)
        """

        saved = 0

        # ----------------------------------------------------
        # Insert rows
        # ----------------------------------------------------

        for _, row in df.iterrows():

            customer_id = row[
                customer_column
            ]

            if pd.isna(customer_id):
                continue

            customer_id = str(
                customer_id
            ).strip()

            if not customer_id:
                continue

            values = (

                customer_id,

                value(
                    row,
                    "gender",
                ),

                value(
                    row,
                    "SeniorCitizen",
                    "senior_citizen",
                ),

                value(
                    row,
                    "Partner",
                    "partner",
                ),

                value(
                    row,
                    "Dependents",
                    "dependents",
                ),

                value(
                    row,
                    "tenure",
                ),

                value(
                    row,
                    "PhoneService",
                    "phone_service",
                ),

                value(
                    row,
                    "MultipleLines",
                    "multiple_lines",
                ),

                value(
                    row,
                    "InternetService",
                    "internet_service",
                ),

                value(
                    row,
                    "OnlineSecurity",
                    "online_security",
                ),

                value(
                    row,
                    "OnlineBackup",
                    "online_backup",
                ),

                value(
                    row,
                    "DeviceProtection",
                    "device_protection",
                ),

                value(
                    row,
                    "TechSupport",
                    "tech_support",
                ),

                value(
                    row,
                    "StreamingTV",
                    "streaming_tv",
                ),

                value(
                    row,
                    "StreamingMovies",
                    "streaming_movies",
                ),

                value(
                    row,
                    "Contract",
                    "contract",
                ),

                value(
                    row,
                    "PaperlessBilling",
                    "paperless_billing",
                ),

                value(
                    row,
                    "PaymentMethod",
                    "payment_method",
                ),

                value(
                    row,
                    "MonthlyCharges",
                    "monthly_charges",
                ),

                value(
                    row,
                    "TotalCharges",
                    "total_charges",
                ),

                value(
                    row,
                    "Churn",
                    "churn",
                ),

                value(
                    row,
                    "Churn Probability",
                    "churn_probability",
                ),

                value(
                    row,
                    "Prediction",
                    "Churn Prediction",
                    "prediction",
                ),

                value(
                    row,
                    "Risk Level",
                    "Customer Risk Segment",
                    "risk_level",
                ),

                value(
                    row,
                    "Customer Risk Score",
                    "Risk Score",
                    "risk_score",
                ),
            )

            try:

                cursor.execute(
                    query,
                    values,
                )

                saved += 1

            except Error as row_error:

                print(
                    f"Row skipped: {row_error}"
                )

        connection.commit()

        print(
            f"Successfully saved {saved} customers."
        )

        return saved

    except Error as exc:

        connection.rollback()

        print(
            f"MySQL save error: {exc}"
        )

        return 0

    finally:

        if cursor is not None:
            cursor.close()

        connection.close()

if __name__ == "__main__":

    print("=" * 55)
    print("ChurnIQ MySQL Connection Test")
    print("=" * 55)

    if test_connection():

        print("✓ MySQL connection successful.")

        if initialize_database():
            print("✓ ChurnIQ database tables ready.")
        else:
            print("✗ Could not create database tables.")

    else:

        print("✗ MySQL connection failed.")

    print("=" * 55)