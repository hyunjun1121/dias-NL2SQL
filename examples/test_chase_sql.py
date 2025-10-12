"""
Simple test to demonstrate CHASE-SQL usage.

Run this to see how CHASE-SQL works with a toy example.
"""

from baseline.chase_sql import CHASESQLBaseline
from utils.llm_client import LLMClient
from utils.database_executor import DatabaseExecutor
import sqlite3
import os


def create_toy_database():
    """Create a simple test database."""
    db_path = "toy_database.db"

    # Remove if exists
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create employees table
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            salary REAL NOT NULL,
            department TEXT NOT NULL,
            hire_date TEXT
        )
    """)

    # Insert sample data
    employees = [
        (1, 'Alice', 75000, 'Engineering', '2020-01-15'),
        (2, 'Bob', 55000, 'Engineering', '2019-06-20'),
        (3, 'Charlie', 45000, 'Engineering', '2021-03-10'),
        (4, 'David', 85000, 'Engineering', '2018-11-05'),
        (5, 'Eve', 65000, 'Marketing', '2020-07-12'),
        (6, 'Frank', 70000, 'Sales', '2019-09-25'),
        (7, 'Grace', 90000, 'Engineering', '2017-04-18'),
        (8, 'Henry', 50000, 'Marketing', '2021-01-30'),
    ]

    cursor.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?)",
        employees
    )

    conn.commit()
    conn.close()

    print(f"✓ Created toy database: {db_path}")
    print(f"  - 8 employees")
    print(f"  - 5 in Engineering (3 with salary > 50000)")
    return db_path


def test_chase_sql():
    """Test CHASE-SQL with toy example."""

    print("="*80)
    print("CHASE-SQL Test")
    print("="*80)

    # Create toy database
    db_path = create_toy_database()

    # Define schema
    schema = {
        "employees": {
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "TEXT"},
                {"name": "salary", "type": "REAL"},
                {"name": "department", "type": "TEXT"},
                {"name": "hire_date", "type": "TEXT"}
            ]
        }
    }

    # Natural language query
    nl_query = "Show employees with salary over 50000 in Engineering department"

    print(f"\n📝 Query: {nl_query}")
    print(f"📊 Schema: employees(id, name, salary, department, hire_date)")

    # Initialize CHASE-SQL
    print(f"\n🔧 Initializing CHASE-SQL...")
    print(f"   Model: gpt-4o (you can change to 'deepseek-r1' for cluster)")

    llm_client = LLMClient(
        model_name="gpt-4o",
        # For cluster, use:
        # model_name="deepseek-r1",
        # base_url="http://your-cluster:8000/v1"
    )
    db_executor = DatabaseExecutor(db_path)
    config = {}

    chase_baseline = CHASESQLBaseline(llm_client, db_executor, config)

    # Generate SQL
    print(f"\n🚀 Running CHASE-SQL (one-shot generation)...")
    result = chase_baseline.generate_sql(nl_query, schema)

    # Display results
    print(f"\n📋 Query Plan (3 steps):")
    for step in result['query_plan']['steps']:
        print(f"   Step {step['step_number']}: {step['step_type']}")
        print(f"      → {step['description']}")

    print(f"\n💻 Generated SQL (one-shot):")
    print(f"   {result['sql']}")

    print(f"\n✨ Execution Result:")
    if result['success']:
        print(f"   ✓ Success!")
        exec_result = result['execution_result']
        print(f"   Rows returned: {exec_result.get('num_rows', 0)}")
        print(f"\n   Sample rows:")
        for i, row in enumerate(exec_result.get('result', [])[:3], 1):
            print(f"      {i}. {row}")
    else:
        print(f"   ✗ Failed")
        print(f"   Error: {result['execution_result'].get('error', 'Unknown')}")

    # Expected result
    print(f"\n🎯 Expected Result:")
    print(f"   Should return 3 employees:")
    print(f"   - Alice (75000)")
    print(f"   - Bob (55000)")
    print(f"   - David (85000)")
    print(f"   - Grace (90000)")

    print(f"\n✅ CHASE-SQL test complete!")
    print(f"\n💡 Key points:")
    print(f"   - One-shot generation (single LLM call)")
    print(f"   - No progressive execution")
    print(f"   - No semantic verification")
    print(f"   - No error recovery")

    # Cleanup
    os.remove(db_path)
    print(f"\n🧹 Cleaned up toy database")


if __name__ == "__main__":
    test_chase_sql()
