import csv
import random
from faker import Faker
from datetime import datetime
import pandas as pd

fake = Faker('en_IN')

# Constants
products = ['Moong Dhall', 'Moong Dust', 'Orid Dhall', 'Orid Dust',
            'Toor Dhall', 'Toor Dust', 'Orid Polish']

expense_types = ['Transport', 'Packing', 'Maintenance', 'Salary', 'Office Supplies']
party_names = ['Sri Traders', 'Balaji & Co', 'Vasanth Stores', 'Ganesh Suppliers', 'Amman Agencies']

# Date ranges
date_range_24_25 = pd.date_range(start='2024-04-01', end='2025-03-31')
date_range_23_24 = pd.date_range(start='2023-04-01', end='2024-03-31')

# Monthly base weights (trending)
monthly_weights = {
    (2023, 4): 0.3, (2023, 5): 0.5, (2023, 6): 0.7, (2023, 7): 1.0,
    (2023, 8): 1.5, (2023, 9): 2.0, (2023, 10): 2.5, (2023, 11): 3.0,
    (2023, 12): 3.5, (2024, 1): 4.0, (2024, 2): 5.0, (2024, 3): 6.0,
    (2024, 4): 7.0, (2024, 5): 8.0, (2024, 6): 9.0, (2024, 7): 10.0,
    (2024, 8): 9.5, (2024, 9): 8.0, (2024, 10): 6.0, (2024, 11): 4.0,
    (2024, 12): 2.5, (2025, 1): 2.0, (2025, 2): 1.5, (2025, 3): 1.0
}

# Get seasonal weight
def get_seasonal_weight(date, mode='sales'):
    month = date.month
    base_weight = monthly_weights.get((date.year, month), 1.0)
    
    if mode == 'sales':
        if month in [4]:  # Chithirai
            return base_weight + 3.0
        elif month in [10, 11]:  # Diwali
            return base_weight + 4.0
        elif month == 1:  # Pongal
            return base_weight + 3.5
    elif mode == 'expense':
        if month in [2, 3]:  # Pre-Pongal
            return base_weight + 3.0
        elif month in [8, 9]:  # Pre-Diwali
            return base_weight + 4.0
        elif month == 3:  # Pre-Chithirai
            return base_weight + 2.0
    return base_weight

def weighted_random_date(date_range, mode='sales'):
    month_pool = []
    for date in date_range:
        weight = get_seasonal_weight(date, mode)
        month_pool.extend([date] * int(weight * 10))
    return random.choice(month_pool).date()

def get_low_margin_rate(cost, margin_factor=0.05):
    margin = random.uniform(margin_factor - 0.02, margin_factor + 0.02)
    return round(cost * (1 + margin), 2)

def get_base_cost(product):
    base_ranges = {
        'Moong Dhall': (3500, 4500),
        'Moong Dust': (3000, 4000),
        'Orid Dhall': (5000, 6500),
        'Orid Dust': (4800, 6200),
        'Toor Dhall': (7000, 9000),
        'Toor Dust': (6000, 8500),
        'Orid Polish': (5200, 7000)
    }
    return random.randint(*base_ranges.get(product, (4000, 7000)))

def get_qty(product):
    return round(random.uniform(0.1, 8 if "Dust" in product else 25), 2)

def get_expense_amount(product, amount, seasonal_factor):
    expense_ratio = random.uniform(0.9, 1.1)
    expense_amount = round(amount * expense_ratio * seasonal_factor, 2)
    return max(min(expense_amount, 100000), 500)

# Generator functions
def generate_expense_data(writer, date_range, rows):
    for _ in range(rows):
        date = weighted_random_date(date_range, mode='expense')
        party = random.choice(party_names)
        product = random.choice(products)
        base_cost = get_base_cost(product)
        expense_type = random.choices(expense_types, weights=[5, 4, 1, 1, 1])[0]
        seasonal_factor = get_seasonal_weight(date, mode='expense')
        sale_price = get_low_margin_rate(base_cost)
        expense_amount = get_expense_amount(product, sale_price, seasonal_factor)
        writer.writerow([date, party, expense_amount, expense_type, product])

def generate_sales_data(writer, date_range, rows):
    for _ in range(rows):
        date = weighted_random_date(date_range, mode='sales')
        party = random.choice(party_names)
        product = random.choice(products)
        qty = get_qty(product)
        cost = get_base_cost(product)
        rate = get_low_margin_rate(cost)
        amount = round(qty * rate, 2)
        total_cost = round(qty * cost, 2)
        profit = round(amount - total_cost, 2)

        if random.random() < 0.05:
            cost *= 1.05
            profit = round(amount - cost * qty, 2)

        writer.writerow([date, party, qty, rate, amount, product, round(cost, 2), profit])

# Batch configuration
batches = 6
rows_per_batch = 50000 // batches  # ~8333 per batch
date_ranges = [date_range_24_25] * 3 + [date_range_23_24] * 3

# Generate expenses.csv
with open('expenses.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Party Name', 'Amount', 'Expense Type', 'Product'])
    for i in range(batches):
        generate_expense_data(writer, date_ranges[i], rows_per_batch)

# Generate product-sales.csv
with open('product-sales.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Party Name', 'Qty', 'Rate', 'Amount', 'Product', 'Cost', 'Profit'])
    for i in range(batches):
        generate_sales_data(writer, date_ranges[i], rows_per_batch)
