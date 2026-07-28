# ============================================================
# MARKET DESCRIPTIVE STATISTICS + ADF TESTS
# Spyder 6 / macOS / paper_spyder environment
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller


# ============================================================
# 1. DATA LOCATION
# ============================================================

DATA_FOLDER = Path("/Users/meriem.fereva/Desktop/paper")


# ============================================================
# 2. FILE NAMES
# ============================================================

FILE_NAMES = {
    "S&P500": "SP500_data.csv",
    "NASDAQ100": "NASDAQ100_data.csv",
    "DowJones": "DOWJONES_data.csv",
    "Russell2000": "RUSSELL2000_data.csv",
}

MARKET_ORDER = [
    "S&P500",
    "NASDAQ100",
    "DowJones",
    "Russell2000",
]


# ============================================================
# 3. SIGNIFICANCE STARS
# ============================================================

def get_sig_symbol(p_value):
    """
    Conventional significance notation:
        ***  p < 0.01
        **   p < 0.05
        *    p < 0.10
    """

    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    else:
        return ""


# ============================================================
# 4. LOAD CSV FILES
# ============================================================

def load_market_data():
    """
    Load all four market CSV files.

    Required original column:
        price

    Optional date column:
        Time
    """

    market_data = {}

    print("=" * 75)
    print("LOADING MARKET DATA")
    print("=" * 75)

    for market_name, file_name in FILE_NAMES.items():

        file_path = DATA_FOLDER / file_name

        print(f"\nLoading {market_name}")
        print(f"File: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(
                f"\nCould not find the file:\n{file_path}\n\n"
                "Make sure the CSV file is stored in the DATA_FOLDER."
            )

        df = pd.read_csv(file_path)

        # Remove accidental spaces from column names
        df.columns = df.columns.str.strip()

        print(f"Columns found: {list(df.columns)}")

        # Check required price column
        if "price" not in df.columns:
            raise ValueError(
                f"\n'{file_name}' does not contain a column named 'price'.\n"
                f"Columns found: {list(df.columns)}"
            )

        # Convert price column to numeric
        df["price"] = pd.to_numeric(
            df["price"],
            errors="coerce"
        )

        # Convert Time column if present
        if "Time" in df.columns:

            df["Time"] = pd.to_datetime(
                df["Time"],
                errors="coerce"
            )

            # Sort chronologically
            df = (
                df.sort_values("Time")
                .reset_index(drop=True)
            )

        market_data[market_name] = df

        print(
            f"{market_name} loaded successfully "
            f"with {len(df):,} rows."
        )

    return market_data


# ============================================================
# 5. PREPARE RETURNS AND TARGET
# ============================================================

def prepare_market(df):
    """
    Create log returns and the volatility target.

    return:
        log(price_t) - log(price_t-1)

    target:
        5-day rolling standard deviation of returns,
        shifted one observation forward.

    Note:
    This preserves the target definition from your original code.
    """

    df = df.copy()

    # Logarithms require strictly positive prices
    df.loc[df["price"] <= 0, "price"] = np.nan

    # Daily log returns
    df["return"] = np.log(df["price"]).diff()

    # Volatility target
    df["target"] = (
        df["return"]
        .rolling(window=5)
        .std()
        .shift(-1)
    )

    # Clean possible infinite values
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return df


# ============================================================
# 6. LOAD AND PREPARE ALL MARKETS
# ============================================================

market_data = load_market_data()

markets = {
    name: prepare_market(df)
    for name, df in market_data.items()
}


# ============================================================
# 7. BASIC DATA CHECK
# ============================================================

print("\n")
print("=" * 75)
print("DATA CHECK")
print("=" * 75)

for name in MARKET_ORDER:

    df = markets[name]

    print(f"\n{name}")

    columns_to_show = [
        col
        for col in [
            "Time",
            "price",
            "return",
            "target"
        ]
        if col in df.columns
    ]

    print(df[columns_to_show].head())


# ============================================================
# 8. TABLE 1 — DESCRIPTIVE STATISTICS
# ============================================================

print("\n")
print("=" * 75)
print("TABLE 1 — DESCRIPTIVE STATISTICS")
print("=" * 75)


stats_rows = [
    "Observations",
    "Mean",
    "Median",
    "Maximum",
    "Minimum",
    "Std. Deviation",
    "Skewness",
    "Kurtosis",
    "Jarque-Bera"
]


for row in stats_rows:

    cells = [row]

    for name in MARKET_ORDER:

        r = (
            markets[name]["return"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if row == "Observations":

            result = f"{len(r):,}"

        elif row == "Mean":

            result = f"{r.mean():.4f}"

        elif row == "Median":

            result = f"{r.median():.4f}"

        elif row == "Maximum":

            result = f"{r.max():.4f}"

        elif row == "Minimum":

            result = f"{r.min():.4f}"

        elif row == "Std. Deviation":

            result = f"{r.std(ddof=1):.4f}"

        elif row == "Skewness":

            result = f"{skew(r, bias=False):.4f}"

        elif row == "Kurtosis":

            result = (
                f"{kurtosis(r, fisher=False, bias=False):.4f}"
            )

        elif row == "Jarque-Bera":

            jb_result = jarque_bera(r)

            jb_stat = jb_result.statistic
            jb_p = jb_result.pvalue

            result = (
                f"{jb_stat:.2f}"
                f"{get_sig_symbol(jb_p)}"
            )

        cells.append(result)

    # LaTeX-ready formatting
    print(
        " & ".join(cells)
        + " \\\\"
    )

# ============================================================
# 9. TABLE 2 — ADF TESTS ON PRICE LEVELS
# ============================================================

print("\n")
print("=" * 75)
print("TABLE 2 — AUGMENTED DICKEY-FULLER TESTS")
print("=" * 75)

print("\nPRICE LEVELS")
print("-" * 75)


price_adf_results = {}


for name in MARKET_ORDER:

    price_series = (
        markets[name]["price"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(price_series) < 20:
        raise ValueError(
            f"Not enough valid observations for ADF test: {name}"
        )

    result = adfuller(
        price_series,
        regression="c",
        autolag="AIC"
    )

    price_adf_results[name] = result

    adf_stat = result[0]
    p_value = result[1]

    print(
        f"Raw {name}"
        f" & {adf_stat:.3f}"
        f" & {p_value:.4f}"
        f"{get_sig_symbol(p_value)}"
        f" \\\\"
    )


# ============================================================
# 10. ADF TESTS ON RETURNS
# ============================================================

print("\\addlinespace")

print("\nRETURNS")
print("-" * 75)


return_adf_results = {}


for name in MARKET_ORDER:

    return_series = (
        markets[name]["return"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(return_series) < 20:
        raise ValueError(
            f"Not enough valid observations for ADF test: {name}"
        )

    result = adfuller(
        return_series,
        regression="c",
        autolag="AIC"
    )

    return_adf_results[name] = result

    adf_stat = result[0]
    p_value = result[1]

    print(
        f"{name} Returns"
        f" & {adf_stat:.3f}"
        f" & {p_value:.4f}"
        f"{get_sig_symbol(p_value)}"
        f" \\\\"
    )


# ============================================================
# 11. ADF CRITICAL VALUES
# ============================================================

critical_values = (
    return_adf_results["Russell2000"][4]
)

print("\\midrule")

print(
    "Critical Values: "
    f"1%: {critical_values['1%']:.2f} | "
    f"5%: {critical_values['5%']:.2f} | "
    f"10%: {critical_values['10%']:.2f}"
)


# ============================================================
# 12. OPTIONAL: CREATE PANDAS TABLE OF DESCRIPTIVE STATS
# ============================================================

descriptive_results = []

for name in MARKET_ORDER:

    r = (
        markets[name]["return"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    jb_result = jarque_bera(r)

    descriptive_results.append(
        {
            "Market": name,
            "Observations": len(r),
            "Mean": r.mean(),
            "Median": r.median(),
            "Maximum": r.max(),
            "Minimum": r.min(),
            "Std. Deviation": r.std(ddof=1),
            "Skewness": skew(r, bias=False),
            "Kurtosis": kurtosis(
                r,
                fisher=False,
                bias=False
            ),
            "Jarque-Bera": jb_result.statistic,
            "JB p-value": jb_result.pvalue,
        }
    )


descriptive_table = pd.DataFrame(
    descriptive_results
)

descriptive_table = descriptive_table.set_index(
    "Market"
)


print("\n")
print("=" * 75)
print("DESCRIPTIVE STATISTICS AS DATAFRAME")
print("=" * 75)

print(
    descriptive_table.round(4)
)


# ============================================================
# 13. OPTIONAL: CREATE PANDAS TABLE OF ADF RESULTS
# ============================================================

adf_results = []

for name in MARKET_ORDER:

    price_result = price_adf_results[name]
    return_result = return_adf_results[name]

    adf_results.append(
        {
            "Market": name,
            "Price ADF Statistic": price_result[0],
            "Price p-value": price_result[1],
            "Return ADF Statistic": return_result[0],
            "Return p-value": return_result[1],
        }
    )


adf_table = pd.DataFrame(
    adf_results
)

adf_table = adf_table.set_index(
    "Market"
)


print("\n")
print("=" * 75)
print("ADF RESULTS AS DATAFRAME")
print("=" * 75)

print(
    adf_table.round(4)
)


# ============================================================
# 14. OPTIONAL: SAVE RESULTS TO CSV
# ============================================================

OUTPUT_FOLDER = DATA_FOLDER / "output"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


descriptive_output = (
    OUTPUT_FOLDER
    / "descriptive_statistics.csv"
)

adf_output = (
    OUTPUT_FOLDER
    / "adf_results.csv"
)


descriptive_table.to_csv(
    descriptive_output
)

adf_table.to_csv(
    adf_output
)


print("\n")
print("=" * 75)
print("OUTPUT FILES SAVED")
print("=" * 75)

print(
    f"Descriptive statistics:\n"
    f"{descriptive_output}"
)

print(
    f"\nADF results:\n"
    f"{adf_output}"
)


# ============================================================
# 15. FINISHED
# ============================================================

print("\n")
print("=" * 75)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 75)