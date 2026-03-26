# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 09:58:39 2026

@author: FAZA
"""

# =========================================
# Step 0: Setup
# =========================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. LOAD RAW DATA
# =========================================================

raw_menu_df = pd.read_csv("Pizza Company Dataset.csv", sep=";")

# =========================================
# Data completeness check
# =========================================

missing_summary = (
    raw_menu_df
    .isna()
    .mean()
    .reset_index()
    .rename(columns={"index": "column", 0: "missing_pct"})
)

missing_summary["missing_pct"] = missing_summary["missing_pct"] * 100

missing_summary.sort_values("missing_pct", ascending=False)

# check raw duplicates (same exact menu name)

raw_dupes = (
    raw_menu_df.groupby(["id", "menus.name"])
      .size()
      .reset_index(name="row_count")
      .query("row_count > 1")
)

# =========================================================
# 2. TEXT CLEANING
# =========================================================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

raw_menu_df["clean_name"] = raw_menu_df["menus.name"].apply(clean_text)

# Menu Name Frequency Analysis
top20 = (
    raw_menu_df.groupby("clean_name")["name"]
      .nunique()
      .sort_values(ascending=False)
      .head(20)
      .reset_index()
      .rename(columns={"name": "restaurant_count"})
)

plt.figure(figsize=(9, 6))
sns.barplot(
    data=top20,
    y="clean_name",
    x="restaurant_count"
)
plt.xlabel("Number of Restaurants")
plt.ylabel("Pizza Name")
plt.title("Top 20 Most Common Pizza Items")
plt.tight_layout()
plt.show()

# =========================================================
# 3. REMOVE NON-DIFFERENTIATING MODIFIERS
# =========================================================

NOISE_WORDS = [
    r"\bsmall\b", r"\bmedium\b", r"\blarge\b", r"\bxl\b",
    r"\bpersonal\b", r"\bfamily\b", r"\bgiant\b", r"\bparty\b",
    r"\bclassic\b", r"\bplain\b", r"\boriginal\b", r"\bregular\b",
    r"\btraditional\b", r"\bthin crust\b", r"\bpan\b"
]

def remove_noise(text):
    for w in NOISE_WORDS:
        text = re.sub(w, "", text)
    return re.sub(r"\s+", " ", text).strip()

raw_menu_df["base_name"] = raw_menu_df["clean_name"].apply(remove_noise)

# =========================================================
# 4. CANONICAL MENU MAPPING
# =========================================================

def canonicalize(name):
    if "slice" in name:
        return "Pizza Slice"
    if "philly" in name:
        return "Philly Cheese Steak Pizza"
    if "deep" in name:
        return "Chicago Deep Dish Pizza"
    if "steak" in name:
        return "Steak Pizza"
    if "hawaiian" in name or "pineapple" in name:
        return "Hawaiian Pizza"
    if "bbq" in name and "chicken" in name:
        return "BBQ Chicken Pizza"
    if "buffalo" in name:
        return "Buffalo Chicken Pizza"
    if "meat lover" in name or "meat lovers" in name:
        return "Meat Lovers Pizza"
    if "supreme" in name or "deluxe" in name or "works" in name:
        return "Supreme Pizza"
    if "taco" in name:
        return "Taco Pizza"
    if "breakfast" in name:
        return "Breakfast Pizza"
    if "pepperoni" in name:
        return "Pepperoni Pizza"
    if "margherita" in name or "margarita" in name:
        return "Margherita Pizza"
    if "white" in name or "bianca" in name:
        return "White Pizza"
    if "veggie" in name or "vegetarian" in name or "garden" in name:
        return "Veggie Pizza"
    if name in ["cheese pizza", "cheese"]:
        return "Cheese Pizza"
    return name.title()

raw_menu_df["canonical_name"] = raw_menu_df["base_name"].apply(canonicalize)

# =========================================================
# 5. CORE VS NOVELTY CLASSIFICATION
# =========================================================

CORE_SKUS = {
    "Cheese Pizza",
    "Pepperoni Pizza",
    "Margherita Pizza",
    "Veggie Pizza",
    "White Pizza"
}

raw_menu_df["menu_type"] = np.where(
    raw_menu_df["canonical_name"].isin(CORE_SKUS),
    "Core",
    "Novelty"
)

# =========================================================
# 6. BUILD CANONICAL MENU TABLE (MAIN DATASET)
# Grain: restaurant_id × canonical_menu
# =========================================================

menu_canonical_df = (
    raw_menu_df
    .groupby(["id", "name", "canonical_name", "menu_type"])
    .agg(
        typical_price=("menus.amountMin", "median"),
        n_price_points=("menus.amountMin", "nunique")
    )
    .reset_index()
)

menu_canonical_df["has_tiers"] = menu_canonical_df["n_price_points"] > 1

# =========================================================
# 7. ATTACH RESTAURANT ATTRIBUTES + CLEAN PROVINCE
# =========================================================

restaurant_attrs = (
    raw_menu_df[["id", "city", "province", "review_overall"]]
    .drop_duplicates(subset=["id"])
)

menu_canonical_df = menu_canonical_df.merge(
    restaurant_attrs,
    on="id",
    how="left"
)

# ------------------------------
# Province remap dictionary
# ------------------------------
# Province Analysis
# Count how many province in the data
raw_menu_df["province"].nunique() #281 

# See non acrynom provinces
non_acronym_province_with_city = (
    raw_menu_df.loc[
        raw_menu_df["province"].notna() &
        (raw_menu_df["province"].str.len() > 2),
        ["province", "city"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

non_acronym_province_with_city

# Remap Non Acronym Provinces into States

PROVINCE_REMAP = {
    "Alt De Berwind": "PR",
    "Arco-plaza": "CA",
    "Arnold": "NC",
    "Atl": "GA",
    "Bamboo": "NC",
    "Bammel": "TX",
    "Baxter Estates": "NY",
    "Bellefonte": "KY",
    "Bengal": "LA",
    "Bicentennial": "CA",
    "Bloomfld Hls": "MI",
    "Bloomington Heights": "IL",
    "Bloomington Hills": "UT",
    "Blue Anchor": "NJ",
    "Bonney Lk": "WA",
    "Bouquet Canyon": "CA",
    "Brandtsville": "PA",
    "Brentwood": "CA",
    "Briarcliff Mnr": "NY",
    "Brownhelm": "OH",
    "Brownsboro Farm": "KY",
    "Brownstown Twp": "MI",
    "Bunker Hill Village": "TX",
    "Burlngtn City": "VT",
    "Camden": "NJ",
    "Camroden": "NY",
    "Carpolis": "PA",
    "Cascade": "MI",
    "Cherry Hill Township": "NJ",
    "Cherry Hl Vlg": "NJ",
    "Chevy Chs Vlg": "MD",
    "Choconut Center": "NY",
    "Chupadero": "NM",
    "Cimarron Hills": "CO",
    "City Of Spokane Valley": "WA",
    "City Of Wb": "PA",
    "Clarkson Valley": "MO",
    "Clover Bottom": "MO",
    "Co Spgs": "CO",
    "Collings Lakes": "NJ",
    "Colorado Spgs": "CO",
    "Country Life Acres": "MO",
    "Cresthill": "IL",
    "Crestview Heights": "NY",
    "Cushing": "UT",
    "Dania Beach": "FL",
    "Darrtown": "OH",
    "Davidsburg": "PA",
    "Deephaven": "MN",
    "Deer Wells": "NY",
    "Deerfield Beach": "FL",
    "Derry Church": "PA",
    "Drexelbrook": "PA",
    "Durlach": "PA",
    "East Haven": "VA",
    "East Htfd": "CT",
    "Echelon": "NJ",
    "Edmund": "SC",
    "Elmhurst": "NY",
    "Elnora": "NY",
    "Evesham Twp": "NJ",
    "Fair Haven": "CT",
    "Fairmont": "OH",
    "Fayson Lakes": "NJ",
    "Fire Island Pines": "NY",
    "Fmy": "FL",
    "Forest View": "IL",
    "Fort Dearborn": "IL",
    "Fort Devens": "MA",
    "Fox Rv Vly Gn": "IL",
    "Framingham So": "MA",
    "Fredon Twp": "NJ",
    "Friendsville": "IL",
    "Fruitville": "PA",
    "Ft Lauderdale": "FL",
    "G P O": "NY",
    "Gilgo Beach": "NY",
    "Glendale": "WI",
    "Gloucester City": "NJ",
    "Great Notch": "NJ",
    "Groesbeck": "OH",
    "Grove": "NJ",
    "Guilford Courthouse National": "NC",
    "Hamilton Twp": "NJ",
    "Hanes": "NC",
    "Harristown": "PA",
    "Hassan": "MN",
    "Hbg Inter Airp": "PA",
    "Holly Ridge": "LA",
    "Hollywood Park": "TX",
    "Honey Creek": "MO",
    "Hono": "HI",
    "Howard": "WI",
    "Htfd": "CT",
    "Huetter": "ID",
    "Islip Manor": "NY",
    "Jackson Hole": "WY",
    "Joppatowne": "MD",
    "Juanita": "WA",
    "Kingsgate": "WA",
    "La Conchita": "CA",
    "Lake Mahopac": "NY",
    "Larimers Corner": "WA",
    "Lawrenceville": "NJ",
    "Longview": "NC",
    "Los Feliz": "CA",
    "Macomb Twp": "MI",
    "Magnolia": "WA",
    "Malba": "NY",
    "Manhattan": "NY",
    "Manhattanville": "NY",
    "Manor Ridge": "PA",
    "Marble Cliff": "OH",
    "Margate": "NJ",
    "Matthewstown": "NC",
    "Mayfield Hts": "OH",
    "Mckownville": "NY",
    "Miami": "FL",
    "Midtown": "NJ",
    "Montville Township": "NJ",
    "Mormon Island": "CA",
    "Mount Laurel Township": "NJ",
    "Mount Ross": "NY",
    "Mozart": "WV",
    "Mphs": "TN",
    "Murdock": "CO",
    "Murfreesbr": "TN",
    "N Egremont": "MA",
    "N Haven": "CT",
    "Naples": "CA",
    "New York City": "NY",
    "Nlr": "AR",
    "No Bethesda": "MD",
    "No Natick": "MA",
    "No Providence": "RI",
    "Norland Park": "IN",
    "North Glenn": "CO",
    "North Redington Beach": "FL",
    "North White Plains": "NY",
    "Nyc": "NY",
    "Ny": "NY",
    "Oak Park Hts": "MN",
    "Oella": "MD",
    "Oldtown": "NC",
    "Olivenhain": "CA",
    "Ontario Street": "IL",
    "Palmer": "IN",
    "Pamrapo": "NJ",
    "Paradise Park": "CA",
    "Park Hills": "KY",
    "Pembroke Pnes": "FL",
    "Phila": "PA",
    "Pitt": "PA",
    "Plantation": "FL",
    "Pleasant Valley": "NV",
    "Pleasantdale": "NY",
    "Pls Vrds Est": "CA",
    "Prt Jefferson": "NY",
    "Queen City": "VT",
    "Queens": "NY",
    "Queensgate": "WA",
    "Quincy Center": "MA",
    "Radburn": "NJ",
    "Rahns": "PA",
    "Randallsville": "OH",
    "Raugust": "WA",
    "Richmond Highlands": "WA",
    "Ritz": "NJ",
    "Riverchase": "AL",
    "Riverton": "VA",
    "Rivervale": "NJ",
    "Rockville": "CT",
    "Rockville Center": "NY",
    "Round Lake Heights": "IL",
    "Ruscombmanor Twp": "PA",
    "S Connelsvl": "PA",
    "Saint Andrews": "SC",
    "Saint Anne": "MO",
    "Saint Davids": "PA",
    "Saint Pete Beach": "FL",
    "Salt Lake Cty": "UT",
    "San Marin": "CA",
    "Santa Venetia": "CA",
    "Seabrook Island": "SC",
    "Seacliff": "CA",
    "Shawnee Mission": "KS",
    "Sheffield Village": "OH",
    "Sky Valley": "CA",
    "So Portland": "ME",
    "So Yarmouth": "MA",
    "South Bowie": "MD",
    "South St Paul": "MN",
    "Spuyten Duyvil": "NY",
    "Ssf": "CA",
    "St Albans": "VT",
    "St Remy": "NY",
    "Strawberry Point": "CA",
    "Sunrise": "FL",
    "Syr": "NY",
    "Talladega Springs": "AL",
    "Togus": "ME",
    "Townley": "NJ",
    "Uptown": "NJ",
    "Valley View": "IL",
    "Valleyview": "OH",
    "Venetian Islands": "FL",
    "Village Of Mastic Beach": "NY",
    "Village Of Wellington": "FL",
    "Vlg Of 4 Ssns": "MO",
    "Wankers Corners": "OR",
    "Wapping": "CT",
    "Washington Township": "OH",
    "Wedgwood": "WA",
    "Weirs Beach": "NH",
    "Wesley Chapel": "NC",
    "West Deerfield": "IL",
    "West Fort Lee": "NJ",
    "West Glenville": "NY",
    "West Medford": "MA",
    "West Mifflin": "PA",
    "West Pittsburg": "PA",
    "West Vail": "CO",
    "Weymouth Nas": "MA",
    "Wheatfield": "NY",
    "Williams Crk": "IN",
    "Willoughby Hills": "OH",
    "Wilm": "DE",
    "Wilton Manors": "FL",
    "Wla": "CA",
    "Wm Penn Anx W": "PA",
    "Woodbury": "NJ",
}

menu_canonical_df["province_clean"] = (
    menu_canonical_df["province"]
    .replace(PROVINCE_REMAP)
)

raw_menu_df["province_clean"] = (
    raw_menu_df["province"]
    .replace(PROVINCE_REMAP)
)

# =========================================================
# 8. RESTAURANT-LEVEL SUMMARY
# =========================================================

restaurant_ratings = (
    raw_menu_df
    .groupby("id")
    .agg(
        avg_review=("review_overall", "mean")
    )
    .reset_index()
)

restaurant_summary_df = (
    menu_canonical_df
    .groupby(["id", "name", "city", "province_clean"])
    .agg(
        total_menus=("canonical_name", "nunique"),
        core_menus=("menu_type", lambda x: (x == "Core").sum()),
        novelty_menus=("menu_type", lambda x: (x == "Novelty").sum()),
        tiered_menus=("has_tiers", "sum"),
        typical_menu_price=("typical_price", "median")
    )
    .reset_index()
)

restaurant_summary_df = restaurant_summary_df.merge(
    restaurant_ratings,
    on="id",
    how="left"
)

# =========================================================
# 9. CANONICAL MENU FREQUENCY (PLOT)
# =========================================================

menu_frequency = (
    menu_canonical_df
    .groupby("canonical_name")["id"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="restaurant_count")
)

menu_frequency_plot = menu_frequency.head(15).copy()

menu_frequency_plot["menu_group"] = np.where(
    menu_frequency_plot["canonical_name"].isin(CORE_SKUS),
    "Core pizzas",
    "Other pizzas"
)

plt.figure(figsize=(9, 6))

ax = sns.barplot(
    data=menu_frequency_plot,
    x="restaurant_count",
    y="canonical_name",
    hue="menu_group",
    dodge=False,   # keeps bars aligned, not side-by-side
    palette={
        "Core pizzas": "#2C3E50",   # dark, serious
        "Other pizzas": "#BDC3C7"   # muted grey
    }
)

plt.title("Menu Demand Concentrates on a Small Set of Core Pizzas")
plt.xlabel("Number of Restaurants Offering the Item")
plt.ylabel("Canonical Menu")

# Clean legend
ax.legend(title="", loc="lower right")

plt.tight_layout()
plt.show()

# =========================================================
# 10. CORE VS NOVELTY PENETRATION (RESTAURANT LEVEL)
# =========================================================

# Top 5 canonical pizzas by number of restaurants
top_5_canonical = (
    menu_canonical_df
    .groupby("canonical_name")["id"]
    .nunique()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)

# Total unique restaurants
total_restaurants = menu_canonical_df["id"].nunique()

# Restaurants that offer at least one of the top 5 pizzas
restaurants_with_top5 = (
    menu_canonical_df[
        menu_canonical_df["canonical_name"].isin(top_5_canonical)
    ]["id"]
    .nunique()
)

# Percentage
top5_restaurant_share = restaurants_with_top5 / total_restaurants * 100

print(
    f"Top 5 canonical pizzas appear in approximately "
    f"{top5_restaurant_share:.1f}% of restaurants."
)

# =========================================================
# 11. CHAIN VS INDEPENDENT CLASSIFICATION
# =========================================================

restaurant_summary_df["location_count"] = (
    restaurant_summary_df
    .groupby("name")["id"]
    .transform("nunique")
)

restaurant_summary_df["restaurant_type"] = np.where(
    restaurant_summary_df["location_count"] > 1,
    "Chain",
    "Independent"
)

# =========================================================
# 12. CHAIN VS INDEPENDENT SHARE
# =========================================================

chain_split = (
    restaurant_summary_df["restaurant_type"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
    .rename(columns={"index": "restaurant_type", "restaurant_type": "share_pct"})
)

print(chain_split)

# =========================================================
# 10. PRICE OUTLIER DETECTION (QUESTION 2)
# =========================================================

outlier_df = (
    menu_canonical_df
    .dropna(subset=["typical_price", "province_clean"])
    .copy()
)

excluded_count = menu_canonical_df["typical_price"].isna().sum()
print(f"Menus excluded due to missing price: {excluded_count}")

province_price_stats = (
    outlier_df
    .groupby("province_clean")["typical_price"]
    .quantile([0.25, 0.75])
    .unstack()
    .reset_index()
    .rename(columns={0.25: "Q1", 0.75: "Q3"})
)

province_price_stats["IQR"] = province_price_stats["Q3"] - province_price_stats["Q1"]
province_price_stats["lower_bound"] = province_price_stats["Q1"] - 1.5 * province_price_stats["IQR"]
province_price_stats["upper_bound"] = province_price_stats["Q3"] + 1.5 * province_price_stats["IQR"]

outlier_df = outlier_df.merge(
    province_price_stats,
    on="province_clean",
    how="left"
)

outlier_df["price_outlier_flag"] = np.where(
    (outlier_df["typical_price"] > outlier_df["upper_bound"]) |
    (outlier_df["typical_price"] < outlier_df["lower_bound"]),
    "outlier",
    "normal"
)

overall_outlier_pct = (outlier_df["price_outlier_flag"] == "outlier").mean()
print(f"Overall share of menu items flagged as price outliers: {overall_outlier_pct:.2%}")

province_outlier_summary = (
    outlier_df
    .groupby("province_clean")["price_outlier_flag"]
    .value_counts(normalize=True)
    .rename("pct")
    .reset_index()
    .query("price_outlier_flag == 'outlier'")
    .sort_values("pct", ascending=False)
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=province_outlier_summary,
    x="pct",
    y="province_clean"
)
plt.xlabel("Share of Menu Items Flagged as Price Outliers")
plt.ylabel("Province")
plt.title("Price Outlier Concentration by Province")
plt.tight_layout()
plt.show()

# =========================================================
# 10B. ATTACH PRICE OUTLIER FLAG BACK TO MENU_CANONICAL_DF
# =========================================================

menu_canonical_df = menu_canonical_df.merge(
    outlier_df[
        ["id", "canonical_name", "province_clean", "price_outlier_flag"]
    ],
    on=["id", "canonical_name", "province_clean"],
    how="left"
)

# Fill missing flags (menus without valid price)
menu_canonical_df["price_outlier_flag"] = (
    menu_canonical_df["price_outlier_flag"]
    .fillna("missing_price")
)

# =========================================================
# 11. EXPORT CLEAN DATASETS
# =========================================================

raw_menu_df.to_excel("raw_menu_clean.xlsx", index=False)
menu_canonical_df.to_excel("menu_canonical_clean.xlsx", index=False)
restaurant_summary_df.to_csv("restaurant_summary_clean.csv", index=False)

print("Pipeline completed successfully.")
print("- raw_menu_clean.xlsx")
print("- menu_canonical_clean.xlsx  (SQL Question 2 input)")
print("- restaurant_summary_clean.csv  (Power BI / summary)")

### XXX Case Study Analysis

# =========================================
# DOES RESTAURANT CATEGORY PLAY A ROLE?
# =========================================

restaurant_categories = (
    raw_menu_df[["id", "categories"]]
    .dropna(subset=["categories"])
    .assign(
        primary_category=lambda x: x["categories"].str.split(",").str[0].str.strip()
    )
    .drop_duplicates(subset=["id"])   # one row per restaurant
    ["primary_category"]
    .value_counts()
    .reset_index()
)

restaurant_categories.columns = ["Primary Category", "Restaurant Count"]

restaurant_categories

# Plot the restaurant categories count
top_n = 15
plot_df = restaurant_categories.head(top_n)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=plot_df,
    x="Restaurant Count",
    y="Primary Category"
)

plt.title("Primary Restaurant Category Distribution")
plt.xlabel("Number of Restaurants")
plt.ylabel("Primary Category")
plt.tight_layout()
plt.show()

# =========================================
# PRIMARY CATEGORY (FIRST TAG ONLY)
# =========================================

primary_category_df = (
    raw_menu_df[["id", "categories"]]
    .dropna(subset=["categories"])
    .assign(
        primary_category=lambda x: x["categories"].str.split(",").str[0].str.strip()
    )
    .drop_duplicates(subset=["id"])
)

primary_category_df.head()

# =========================================
# JOIN CATEGORY WITH RESTAURANT PRICING
# =========================================

category_price_df = (
    restaurant_summary_df
    .merge(primary_category_df[["id", "primary_category"]], on="id", how="left")
    .dropna(subset=["primary_category", "typical_menu_price"])
)

category_price_df.head()

category_price_summary = (
    category_price_df
    .groupby("primary_category")
    .agg(
        restaurant_count=("id", "nunique"),
        median_price=("typical_menu_price", "median")
    )
    .sort_values("median_price", ascending=False)
    .reset_index()
)

category_price_summary

top_categories = (
    category_price_summary
    .query("restaurant_count >= 30")["primary_category"]
)

plot_df = category_price_df[
    category_price_df["primary_category"].isin(top_categories)
]

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=plot_df,
    x="primary_category",
    y="typical_menu_price"
)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# =========================================================
# 1. MARKET SIZE & STRUCTURE
# =========================================================

sns.set(style="whitegrid")

# Total restaurants
total_restaurants = restaurant_summary_df["id"].nunique()

# Total canonical menu offerings
total_menu_items = menu_canonical_df.shape[0]

print(f"Total pizza restaurants: {total_restaurants}")
print(f"Total canonical menu items: {total_menu_items}")

# Distribution of pizza restaurants per province

import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

state_counts = (
    restaurant_summary_df
    .groupby("province_clean")["id"]
    .nunique()
    .reset_index(name="restaurant_count")
)

fig = px.choropleth(
    state_counts,
    locations="province_clean",
    locationmode="USA-states",
    color="restaurant_count",
    scope="usa",
    color_continuous_scale="Blues",
    title="Distribution of Pizza Restaurants Across US States"
)

fig.show()

# =========================================================
# 2. OVERALL PIZZA PRICE POSITIONING
# (Is pizza cheap / mid / premium?)
# =========================================================

plt.figure(figsize=(8, 5))
sns.histplot(
    restaurant_summary_df["typical_menu_price"].dropna(),
    bins=30,
    kde=True
)

plt.axvline(
    restaurant_summary_df["typical_menu_price"].median(),
    color="red",
    linestyle="--",
    label="Median Price"
)

plt.title("Distribution of Typical Pizza Prices (Restaurant Level)")
plt.xlabel("Typical Pizza Price")
plt.ylabel("Number of Restaurants")
plt.legend()
plt.tight_layout()
plt.show()

print("Portfolio Median Price (Restaurant-weighted):",
      restaurant_summary_df["typical_menu_price"].median())

# =========================================================
# 3. REGIONAL PRICE REALITY
# (Are prices localized or standardized?)
# =========================================================

province_price_summary = (
    restaurant_summary_df
    .dropna(subset=["typical_menu_price", "province_clean"])
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        median_price=("typical_menu_price", "median")
    )
    .reset_index()
    .sort_values("median_price", ascending=False)
)

# Filter provinces with enough observations
province_price_summary = province_price_summary[
    province_price_summary["restaurant_count"] >= 20
]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=province_price_summary,
    x="median_price",
    y="province_clean"
)

plt.title("Median Pizza Price by Province (Sufficient Sample Only)")
plt.xlabel("Median Typical Pizza Price")
plt.ylabel("Province")
plt.tight_layout()
plt.show()

# =========================================================
# 4. CONSUMER EXPECTATION BY REGION
# (Customer rating consistency across provinces)
# =========================================================

rating_df = (
    restaurant_summary_df
    .dropna(subset=["province_clean", "avg_review"])
)

# Top provinces by restaurant count
top_provinces = (
    rating_df["province_clean"]
    .value_counts()
    .head(9)
    .index
)

plot_df = rating_df[
    rating_df["province_clean"].isin(top_provinces)
]

# Overall benchmark (restaurant-level average rating)
overall_avg_rating = restaurant_summary_df["avg_review"].mean()

plt.figure(figsize=(12, 6))

sns.violinplot(
    data=plot_df,
    x="province_clean",
    y="avg_review",
    inner="quartile",
    cut=0,
    color="#4C72B0"  # neutral executive blue
)

# Global benchmark line
plt.axhline(
    overall_avg_rating,
    linestyle="--",
    color="black",
    linewidth=1
)

# Annotate benchmark
plt.text(
    len(top_provinces) - 0.5,
    overall_avg_rating + 0.03,
    f"Overall average rating ≈ {overall_avg_rating:.2f}",
    ha="right",
    fontsize=10,
    color="black"
)

plt.title("Customer Rating Distribution by Province")
plt.xlabel("Province")
plt.ylabel("Customer Rating")
plt.tight_layout()
plt.show()

# =========================================================
# 5. MENU COMPLEXITY — MARKET WIDE
# (How complex are pizza restaurants?)
# =========================================================

menu_counts = restaurant_summary_df["total_menus"]

# Create integer bins centered on whole numbers
max_menus = int(menu_counts.max())
bins = np.arange(0.5, max_menus + 1.5, 1)

plt.figure(figsize=(8, 5))
sns.histplot(
    menu_counts,
    bins=bins,
    kde=True,
    color="#4c72b0"
)

plt.axvline(
    restaurant_summary_df["total_menus"].median(),
    color="red",
    linestyle="--",
    label="Median Menu Count"
)

plt.title("Distribution of Menu Complexity (Menus per Restaurant)")
plt.xlabel("Number of Canonical Menus")
plt.ylabel("Number of Restaurants")
plt.legend()
plt.tight_layout()
plt.show()

print(
    "Median number of menus per restaurant:",
    restaurant_summary_df["total_menus"].median()
)

# Total number of restaurants
total_restaurants = restaurant_summary_df.shape[0]

# Restaurants with 2 or fewer menus
simple_restaurants = restaurant_summary_df[
    restaurant_summary_df["total_menus"] <= 2
].shape[0]

# Percentage
percentage_simple = simple_restaurants / total_restaurants * 100

print(f"{percentage_simple:.1f}% of restaurants list 2 or fewer pizzas")

# =========================================================
# 6. MENU COMPLEXITY BY REGION
# (Does complexity differ geographically?)
# =========================================================

province_complexity_summary = (
    restaurant_summary_df
    .dropna(subset=["province_clean"])
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        avg_menu_count=("total_menus", "mean")
    )
    .reset_index()
)

province_complexity_summary = province_complexity_summary[
    province_complexity_summary["restaurant_count"] >= 20
]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=province_complexity_summary.sort_values("avg_menu_count", ascending=False),
    x="avg_menu_count",
    y="province_clean"
)

plt.title("Average Menu Complexity by Province")
plt.xlabel("Average Number of Menus per Restaurant")
plt.ylabel("Province")
plt.tight_layout()
plt.show()

# =========================================
# MARKET DENSITY vs PRICE (COLOR = RATING)
# =========================================

province_summary = (
    restaurant_summary_df
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        median_price=("typical_menu_price", "median"),
        avg_rating=("avg_review", "mean")
    )
    .reset_index()
)

plot_df = province_summary[
    province_summary["restaurant_count"] >= 20
]

plt.figure(figsize=(10, 6))

ax = sns.scatterplot(
    data=plot_df,
    x="restaurant_count",
    y="median_price",
    hue="avg_rating",
    palette="viridis",        # perceptually uniform
    size=None,                # remove size encoding
    s=300,                    # fixed bubble size
    alpha=0.85
)

# -----------------------------------------
# ADD PROVINCE LABELS
# -----------------------------------------

for _, row in plot_df.iterrows():
    ax.text(
        row["restaurant_count"] + 0.6,
        row["median_price"],
        row["province_clean"],
        fontsize=9,
        alpha=0.9
    )

plt.title("Market Density vs Pricing by Province\n(Color = Average Customer Rating)")
plt.xlabel("Number of Restaurants")
plt.ylabel("Median Typical Pizza Price")
plt.tight_layout()
plt.show()

# XXX

# =========================================================
# HIDDEN VALUE ANALYSIS
# =========================================================

sns.set(style="whitegrid")

# =========================================================
# 1. PROVINCE-LEVEL MARKET PERFORMANCE (BENCHMARKS)
# =========================================================

province_perf = (
    restaurant_summary_df
    .dropna(subset=["province_clean", "typical_menu_price", "avg_review"])
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        median_price=("typical_menu_price", "median"),
        avg_rating=("avg_review", "mean"),
        rating_std=("avg_review", "std")
    )
    .reset_index()
)

# Keep statistically meaningful provinces
province_perf = province_perf[
    province_perf["restaurant_count"] >= 10
]

# =========================================================
# 2. MARKET-WIDE MEDIANS (REFERENCE ONLY)
# =========================================================

# Market benchmark defined as median of province-level medians
# (each province weighted equally to avoid large-state bias)
price_median = province_perf["median_price"].median()
rating_median = province_perf["avg_rating"].median()

print(f"Market median price (Province-normalized): {price_median:.2f}")
print(f"Market median rating (Province-normalized): {rating_median:.2f}")

# =========================================================
# 3. VALUE POCKET CLASSIFICATION (PROVINCE LEVEL)
# =========================================================

def classify_value_pocket(row):
    if row["avg_rating"] >= rating_median and row["median_price"] < price_median:
        return "Under-monetized quality"
    if row["avg_rating"] >= rating_median and row["median_price"] >= price_median:
        return "Defensible premium"
    if row["avg_rating"] < rating_median and row["median_price"] >= price_median:
        return "Overpriced weak quality"
    return "Low-value market"

province_perf["value_pocket"] = province_perf.apply(
    classify_value_pocket, axis=1
)

# =========================================================
# 4. VALUE POCKET SCATTER (ANCHOR CHART)
# =========================================================

plt.figure(figsize=(10, 6))

ax = sns.scatterplot(
    data=province_perf,
    x="median_price",
    y="avg_rating",
    hue="value_pocket",
    size="restaurant_count",
    sizes=(120, 800),
    alpha=0.85,
    palette={
        "Under-monetized quality": "green",
        "Defensible premium": "blue",
        "Overpriced weak quality": "red",
        "Low-value market": "gray"
    }
)

# Benchmark lines
plt.axvline(price_median, linestyle="--", color="black", linewidth=1)
plt.axhline(rating_median, linestyle="--", color="black", linewidth=1)

# Annotate benchmarks directly on chart
plt.text(
    price_median + 0.05,
    province_perf["avg_rating"].min(),
    f"Market price benchmark\n(median of province medians ≈ ${price_median:.2f})",
    fontsize=9,
    ha="left",
    va="bottom"
)

plt.text(
    province_perf["median_price"].min(),
    rating_median + 0.01,
    f"median of province averages ≈ {rating_median:.2f})",
    fontsize=9,
    ha="left",
    va="bottom"
)

# Province labels
for _, r in province_perf.iterrows():
    ax.text(
        r["median_price"] + 0.05,
        r["avg_rating"],
        r["province_clean"],
        fontsize=9,
        alpha=0.9
    )

plt.title("Market Value Pockets: Price vs Customer Rating (Province Level)")
plt.xlabel("Median Pizza Price ($)")
plt.ylabel("Average Customer Rating")
plt.tight_layout()
plt.show()

# =========================================================
# 5. VALUE POCKET MIX (MARKET VIEW)
# =========================================================

value_pocket_mix = (
    province_perf["value_pocket"]
    .value_counts(normalize=True)
    .rename("share")
    .reset_index()
    .rename(columns={"index": "value_pocket"})
)

print(value_pocket_mix)

# =========================================================
# 6. PRIORITY MARKETS (SCREENING)
# =========================================================

priority_markets = (
    province_perf
    .query("value_pocket in ['Under-monetized quality', 'Overpriced weak quality']")
    .sort_values("restaurant_count", ascending=False)
)

priority_markets[[
    "province_clean",
    "restaurant_count",
    "median_price",
    "avg_rating",
    "rating_std",
    "value_pocket"
]]

# =========================================================
# 7. SCALABILITY TEST — RESTAURANT-LEVEL
# =========================================================

# Rename benchmarks explicitly to avoid ambiguity
province_benchmark = province_perf.rename(
    columns={
        "median_price": "province_median_price",
        "avg_rating": "province_avg_rating",
        "restaurant_count": "pricing_eligible_restaurant_count"
    }
)

restaurant_value_flag = (
    restaurant_summary_df
    .merge(
        province_benchmark[
            ["province_clean", "province_median_price", "province_avg_rating"]
        ],
        on="province_clean",
        how="left"
    )
    .dropna(subset=["typical_menu_price", "avg_review"])
)

restaurant_value_flag["under_monetized_flag"] = (
    (restaurant_value_flag["avg_review"] >= restaurant_value_flag["province_avg_rating"]) &
    (restaurant_value_flag["typical_menu_price"] < restaurant_value_flag["province_median_price"])
)

province_scalability = (
    restaurant_value_flag
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        under_monetized_share=("under_monetized_flag", "mean")
    )
    .reset_index()
)

province_scalability = province_scalability[
    province_scalability["restaurant_count"] >= 20
].sort_values("under_monetized_share", ascending=False)

province_scalability

# =========================================================
# 8. PRICING DISCIPLINE DIAGNOSTIC (CORE SKUs)
# =========================================================

core_sku_dispersion = (
    menu_canonical_df
    .query("menu_type == 'Core'")
    .dropna(subset=["typical_price", "province_clean"])
    .groupby(["province_clean", "canonical_name"])
    .agg(
        price_std=("typical_price", "std"),
        restaurant_count=("id", "nunique")
    )
    .reset_index()
)

province_price_discipline = (
    core_sku_dispersion
    .groupby("province_clean")
    .agg(
        avg_core_price_std=("price_std", "mean")
    )
    .reset_index()
)

pricing_vs_value = province_scalability.merge(
    province_price_discipline,
    on="province_clean",
    how="left"
)

pricing_vs_value.sort_values("under_monetized_share", ascending=False)

# =========================================================
# 9. MENU COMPLEXITY vs EXECUTION QUALITY
# =========================================================

province_execution = (
    restaurant_summary_df
    .dropna(subset=["avg_review", "total_menus"])
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        avg_menu_count=("total_menus", "mean"),
        avg_rating=("avg_review", "mean"),
        rating_std=("avg_review", "std")
    )
    .reset_index()
)

province_execution = province_execution.rename(
    columns={
        "rating_std": "execution_rating_std"
    }
)

province_execution = province_execution[
    province_execution["restaurant_count"] >= 20
]

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=province_execution,
    x="avg_menu_count",
    y="avg_rating",
    size="execution_rating_std",
    sizes=(60, 300),
    alpha=0.8
)
plt.xlabel("Average Menu Count")
plt.ylabel("Average Rating")
plt.title("Menu Complexity vs Execution Quality")
plt.tight_layout()
plt.show()

# Correlation between menu complexity and average rating
corr_value = province_execution["avg_menu_count"].corr(
    province_execution["avg_rating"]
)

print(f"Correlation (Menu Complexity vs Avg Rating): {corr_value:.2f}")

# XXX

# =========================================================
# MENU COMPLEXITY vs EXECUTION QUALITY (Province Level)
# =========================================================

province_execution = (
    restaurant_summary_df
    .dropna(subset=["avg_review", "total_menus"])
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        median_menu_count=("total_menus", "median"),
        avg_rating=("avg_review", "mean"),
        rating_std=("avg_review", "std")
    )
    .reset_index()
)

# # Keep statistically meaningful provinces
# province_execution = province_execution[
#     province_execution["restaurant_count"] >= 20
# ]

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=province_execution,
    x="median_menu_count",
    y="avg_rating",
    alpha=0.8,
    s=120
)

# Optional: annotate provinces (lightly)
for _, r in province_execution.iterrows():
    plt.text(
        r["median_menu_count"] + 0.03,
        r["avg_rating"],
        r["province_clean"],
        fontsize=8,
        alpha=0.7
    )

plt.xlabel("Median Menu Count per Restaurant")
plt.ylabel("Average Customer Rating")
plt.title("Menu Complexity vs Execution Quality (Province Level)")
plt.tight_layout()
plt.show()

# Correlation (for reference only)
corr_value = province_execution["median_menu_count"].corr(
    province_execution["avg_rating"]
)

print(f"Correlation (Menu Complexity vs Avg Rating): {corr_value:.2f}")

# 1. Align Aggregation with your "Median" Methodology
province_execution = (
    restaurant_summary_df
    .dropna(subset=["avg_review", "total_menus"])
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        median_menu_count=("total_menus", "median"), # CHANGED from mean to median
        avg_rating=("avg_review", "mean"),
        execution_risk=("avg_review", "std") # Rename for clarity
    )
    .reset_index()
)

# Filter for significance (same as before)
province_execution = province_execution[
    province_execution["restaurant_count"] >= 20
]

# 2. Plot with clearer visual semantics
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=province_execution,
    x="median_menu_count",
    y="avg_rating",
    size="restaurant_count",       # Size = MARKET IMPORTANCE
    sizes=(100, 1000),             # Make the difference obvious
    hue="execution_risk",          # Color = EXECUTION RISK
    palette="viridis_r",           # Reverse palette: Dark/Blue = Low Risk, Yellow/Green = High Risk
    alpha=0.7,
    edgecolor="black"
)

# 3. Add Annotations for Context (Crucial for "Market Reality")
# Annotate the biggest markets or outliers
for i in range(province_execution.shape[0]):
    row = province_execution.iloc[i]
    if row["restaurant_count"] > 50: # Only label big markets to avoid clutter
        plt.text(
            row["median_menu_count"]+0.1, 
            row["avg_rating"], 
            row["province_clean"], 
            fontsize=9,
            weight='bold'
        )

plt.xlabel("Median Menu Count (Complexity)")
plt.ylabel("Average Rating (Execution Quality)")
plt.title("Menu Complexity vs. Execution: Simpler is Often Better")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Market Metrics")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Correlation Check
corr_value = province_execution["median_menu_count"].corr(province_execution["avg_rating"])
print(f"Correlation (Complexity vs Quality): {corr_value:.2f}")

# =========================================================
# 10. DEAL PRIORITIZATION FRAME (FINAL OUTPUT)
# =========================================================

deal_priority = (
    province_perf
    .merge(province_scalability, on="province_clean", how="left")
    .merge(
        province_execution[
            ["province_clean", "avg_menu_count", "execution_rating_std"]
        ],
        on="province_clean",
        how="left"
    )
)

deal_priority = deal_priority[[
    "province_clean",
    "restaurant_count_x",                  # from province_perf (total footprint)
    "median_price",
    "avg_rating",
    "under_monetized_share",
    "avg_menu_count",
    "execution_rating_std",
    "value_pocket"
]].sort_values("restaurant_count_x", ascending=False)

deal_priority

restaurant_value_flag["city_clean"] = (
    restaurant_value_flag["city"]
    .str.lower()
    .str.strip()
)

priority_provinces = ["IL", "TX", "PA", "CA", "NY", "FL"]

# Base dataset
dispersion_base = restaurant_value_flag[
    restaurant_value_flag["province_clean"].isin(priority_provinces)
]

# City-level aggregation
city_summary = (
    dispersion_base
    .groupby(["province_clean", "city_clean"])
    .agg(
        restaurant_count=("id", "nunique"),
        under_monetized_count=("under_monetized_flag", "sum")
    )
    .reset_index()
)

# Rank cities by size within province
city_summary["city_rank"] = (
    city_summary
    .groupby("province_clean")["restaurant_count"]
    .rank(method="first", ascending=False)
)

# ============================
# Province-level dispersion summary
# ============================

dispersion_summary = []

for prov in priority_provinces:
    prov_df = (
        city_summary[city_summary["province_clean"] == prov]
        .sort_values("restaurant_count", ascending=False)
    )

    total_rest = prov_df["restaurant_count"].sum()

    top_1 = prov_df.head(1)
    top_3 = prov_df.head(3)

    dispersion_summary.append({
        "province_clean": prov,
        "total_restaurants": total_rest,
        "top_1_city": top_1["city_clean"].iloc[0],
        "top_1_city_share": round(top_1["restaurant_count"].sum() / total_rest, 2),
        "top_3_cities": ", ".join(top_3["city_clean"].tolist()),
        "top_3_city_share": round(top_3["restaurant_count"].sum() / total_rest, 2),
        "under_monetized_share_in_top_3": round(
            top_3["under_monetized_count"].sum() /
            prov_df["under_monetized_count"].sum(), 2
        ) if prov_df["under_monetized_count"].sum() > 0 else np.nan
    })

dispersion_summary = pd.DataFrame(dispersion_summary)
dispersion_summary

# =========================================
# CORE VS NOVELTY – STRATEGIC MIX ANALYSIS
# =========================================

province_menu_mix = (
    restaurant_summary_df
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        avg_rating=("avg_review", "mean"),
        rating_std=("avg_review", "std"),
        avg_core_menus=("core_menus", "mean"),
        avg_novelty_menus=("novelty_menus", "mean"),
        avg_total_menus=("total_menus", "mean")
    )
    .reset_index()
)

# Focus metric: share of core menus
province_menu_mix["core_menu_share"] = (
    province_menu_mix["avg_core_menus"] /
    province_menu_mix["avg_total_menus"]
)

# Keep meaningful markets
province_menu_mix = province_menu_mix[
    province_menu_mix["restaurant_count"] >= 20
]

plt.figure(figsize=(9, 6))

ax = sns.scatterplot(
    data=province_menu_mix,
    x="core_menu_share",
    y="avg_rating",
    size="rating_std",
    sizes=(100, 600),
    alpha=0.85
)

# Annotate provinces
for _, r in province_menu_mix.iterrows():
    ax.text(
        r["core_menu_share"] + 0.005,
        r["avg_rating"],
        r["province_clean"],
        fontsize=9
    )

plt.xlabel("Core Menu Share (Core / Total Menus)")
plt.ylabel("Average Rating")
plt.title("Menu Focus vs Execution Quality (Province Level)")
plt.tight_layout()
plt.show()

# =========================================
# DISTRIBUTION VIEW: CORE VS NOVELTY BALANCE
# =========================================

# restaurant_summary_df["core_share"] = (
#     restaurant_summary_df["core_menus"] /
#     restaurant_summary_df["total_menus"]
# )

# filtered_restaurants = restaurant_summary_df.merge(
#     province_perf[["province_clean", "value_pocket"]],
#     on="province_clean",
#     how="left"
# )

# filtered_restaurants = filtered_restaurants[
#     filtered_restaurants["province_clean"].isin(["NY", "CA", "FL", "OH", "TX"])
# ]

# plt.figure(figsize=(10, 5))
# sns.stripplot(
#     data=filtered_restaurants,
#     x="province_clean",
#     y="core_share",
#     jitter=0.25,
#     alpha=0.6
# )

# plt.ylabel("Core Menu Share per Restaurant")
# plt.xlabel("Province")
# plt.title("Restaurant-Level Menu Focus Distribution")
# plt.tight_layout()
# plt.show()

# # =========================================
# # CORE VS NOVELTY – BOX PLOT (EXECUTIVE)
# # =========================================

# plot_df = restaurant_summary_df.copy()

# # Core share per restaurant
# plot_df["core_share"] = (
#     plot_df["core_menus"] / plot_df["total_menus"]
# )

# # Drop edge cases
# plot_df = plot_df[
#     plot_df["total_menus"] >= 3
# ]

# # Bin into strategy buckets
# plot_df["menu_strategy"] = pd.cut(
#     plot_df["core_share"],
#     bins=[0, 0.2, 0.35, 1.0],
#     labels=["Novelty-heavy", "Balanced", "Core-focused"]
# )

# plt.figure(figsize=(8, 5))
# sns.boxplot(
#     data=plot_df,
#     x="menu_strategy",
#     y="avg_review",
#     showfliers=False
# )

# plt.xlabel("Menu Strategy")
# plt.ylabel("Average Rating")
# plt.title("Execution Quality by Menu Strategy")
# plt.tight_layout()
# plt.show()


# # =========================================
# # CORE VS NOVELTY – DEFINITIVE PROOF
# # =========================================

# plot_df = restaurant_summary_df.copy()

# # 1. Calculate Share
# plot_df["core_share"] = plot_df["core_menus"] / plot_df["total_menus"]

# # 2. DROP THE FILTER (or keep it only if you specifically want to analyze 'Complex Stores')
# # If you keep the filter, change the title to "Even Among Complex Menus..."
# # Recommended: Remove filter to show the power of the 2-item baseline.
# # plot_df = plot_df[plot_df["total_menus"] >= 2] 

# # 3. Apply the "Mechanical Logic" Bins
# plot_df["menu_strategy"] = pd.cut(
#     plot_df["core_share"],
#     bins=[-0.1, 0.2, 0.4, 1.0], # -0.1 to catch 0.0 values
#     labels=["Novelty-heavy (<20%)", "Drifting (20-35%)", "Core-focused (>35%)"]
# )

# # 4. Plot
# plt.figure(figsize=(9, 6))
# sns.boxplot(
#     data=plot_df,
#     x="menu_strategy",
#     y="avg_review",
#     palette=["#e74c3c", "#f39c12", "#27ae60"], # Red, Orange, Green (Traffic light logic)
#     showfliers=False
# )

# # 5. Add "N" counts to show where the volume is
# # This proves that "Core-focused" is the dominant (and correct) model
# counts = plot_df['menu_strategy'].value_counts()
# for i, label in enumerate(["Novelty-heavy (<20%)", "Drifting (20-35%)", "Core-focused (>35%)"]):
#     plt.text(i, plot_df['avg_review'].min(), f"n={counts[label]}", ha='center', fontweight='bold')

# plt.title("Execution Quality by Menu Strategy", fontsize=14)
# plt.ylabel("Customer Rating (1-5)")
# plt.xlabel("Core Item Share")
# plt.tight_layout()
# plt.show()




# CORE VS NOVELTY – EXECUTION QUALITY (DATA-DRIVEN)
# -----------------------------
# 1. Prepare restaurant-level data
# -----------------------------
plot_df = restaurant_summary_df.copy()

# Core share per restaurant
plot_df["core_share"] = plot_df["core_menus"] / plot_df["total_menus"]

# Replace infinite / invalid values (e.g., total_menus = 0, if any)
plot_df = plot_df.replace([float("inf"), -float("inf")], pd.NA)
plot_df = plot_df.dropna(subset=["core_share", "avg_review"])

# -----------------------------
# 2. Define menu strategy (DATA-DRIVEN, MECHANICAL LOGIC)
# -----------------------------
# Logic:
# - Median menu count = 2 → many restaurants cannot reach 50% core share
# - 0 core items = pure novelty / specialty
# - ~30–35% core share ≈ first deliberate standardization point
# - Above that = clearly core-led

def classify_menu_strategy(x):
    if x == 0:
        return "No core items"
    elif x <= 0.35:
        return "Mixed"
    else:
        return "Core-focused"

plot_df["menu_strategy"] = plot_df["core_share"].apply(classify_menu_strategy)

# -----------------------------
# 3. Fix category order explicitly (VERY IMPORTANT)
# -----------------------------
strategy_order = ["No core items", "Mixed", "Core-focused"]

plot_df["menu_strategy"] = pd.Categorical(
    plot_df["menu_strategy"],
    categories=strategy_order,
    ordered=True
)

# -----------------------------
# 4. Count samples per group (aligned with order)
# -----------------------------
counts = plot_df["menu_strategy"].value_counts().reindex(strategy_order)

# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(9, 6))

ax = sns.boxplot(
    data=plot_df,
    x="menu_strategy",
    y="avg_review",
    order=strategy_order,
    palette=["#95a5a6", "#f39c12", "#27ae60"],  # Grey → Orange → Green
    showfliers=False
)

# -----------------------------
# 6. Annotate sample sizes (bottom)
# -----------------------------
y_min = plot_df["avg_review"].min()

for i, label in enumerate(strategy_order):
    ax.text(
        i,
        y_min,
        f"n={int(counts[label])}",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

# -----------------------------
# 7. Annotate median price per strategy (TOP, CONTEXT ONLY)
# -----------------------------
median_prices = (
    plot_df
    .groupby("menu_strategy")["typical_menu_price"]
    .median()
    .reindex(strategy_order)
)

y_max = plot_df["avg_review"].max()

for i, label in enumerate(strategy_order):
    ax.text(
        i,
        y_max + 0.05,
        f"Median price: ${median_prices[label]:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        alpha=0.85
    )

# -----------------------------
# 8. Labels & formatting
# -----------------------------
plt.title(
    "Execution Quality by Menu Strategy",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("Menu Strategy (Based on Share of Core Items)")
plt.ylabel("Average Customer Rating (1–5)")

plt.ylim(y_min, y_max + 0.2)
plt.tight_layout()
plt.show()





















# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(9, 6))

ax = sns.boxplot(
    data=plot_df,
    x="menu_strategy",
    y="avg_review",
    order=strategy_order,
    palette=["#95a5a6", "#f39c12", "#27ae60"],  # Grey → Orange → Green
    showfliers=False
)

# -----------------------------
# 6. Annotate sample sizes
# -----------------------------
y_min = plot_df["avg_review"].min()

for i, label in enumerate(strategy_order):
    ax.text(
        i,
        y_min,
        f"n={int(counts[label])}",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

# -----------------------------
# 7. Labels & formatting
# -----------------------------
plt.title(
    "Execution Quality by Menu Strategy",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("Menu Strategy (Based on Share of Core Items)")
plt.ylabel("Average Customer Rating (1–5)")

plt.tight_layout()
plt.show()



# Check the distribution first
plot_df["core_share"].describe()

# See where natural breaks occur
plt.figure(figsize=(10, 4))
plt.hist(plot_df["core_share"], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Your 30% threshold')
plt.xlabel("Core Share")
plt.ylabel("Number of Restaurants")
plt.title("Distribution of Core Share Across Portfolio")
plt.legend()
plt.tight_layout()
plt.show()

# Check group sizes with your thresholds
print(plot_df["menu_strategy"].value_counts())
print(f"\nPercentages:")
print(plot_df["menu_strategy"].value_counts(normalize=True) * 100)

# =========================================================
# FINAL VALUE BRIDGE: PRICING HEADROOM TEST
# =========================================================

# 1. Prepare the Data (using existing logic)
rating_threshold = restaurant_summary_df["avg_review"].median()

pricing_headroom_df = (
    restaurant_value_flag
    .query("avg_review >= @rating_threshold")
    .assign(
        monetization_group=lambda x: np.where(
            x["under_monetized_flag"],
            "Under-monetized (High Rating)",
            "Peers (Similar Rating)"
        )
    )
    .dropna(subset=["typical_menu_price"])
)

# 2. Plotting
plt.figure(figsize=(10, 6))

# Use a clean PE palette: Blue for the target opportunity, Grey for the baseline
palette = {"Under-monetized (High Rating)": "#AED6F1", "Peers (Similar Rating)": "#D5DBDB"}

ax = sns.boxplot(
    data=pricing_headroom_df,
    x="monetization_group",
    y="typical_menu_price",
    showfliers=False,
    palette=palette,
    width=0.4
)

# 3. Calculate Medians for the logic anchor
medians = pricing_headroom_df.groupby("monetization_group")["typical_menu_price"].median()
target_median = medians["Under-monetized (High Rating)"]
peer_median = medians["Peers (Similar Rating)"]

# Annotate medians on the boxes
for i, group in enumerate(medians.index):
    ax.text(i, medians[group] + 0.3, f"${medians[group]:.2f}", 
            ha="center", fontsize=11, fontweight="bold", color="#2C3E50")

# 4. ADD THE RISK TIERS (The 'Shading' logic)
# Tier 1: Low Risk (Inflationary pass-through)
plt.axhspan(target_median, target_median + 1.5, color='green', alpha=0.1, label='Tier 1: Low Risk')
# Tier 2: Moderate Risk (Market Alignment)
plt.axhspan(target_median + 1.5, target_median + 3.0, color='orange', alpha=0.1, label='Tier 2: Mod. Risk')
# Tier 3: High Risk (Premium Capture)
plt.axhspan(target_median + 3.0, peer_median, color='red', alpha=0.05, label='Tier 3: High Risk')

# 5. Annotate Tier Labels on the left
plt.text(-0.45, target_median + 0.5, "Tier 1: Low Risk\n(Inflationary)", color='green', fontweight='bold', fontsize=9)
plt.text(-0.45, target_median + 2.1, "Tier 2: Mod. Risk\n(Market Align)", color='darkorange', fontweight='bold', fontsize=9)
plt.text(-0.45, target_median + 3.6, "Tier 3: High Risk\n(Premium)", color='red', fontweight='bold', fontsize=9)

# 6. Final Formatting
plt.title(f"Commercial Health: Pricing Headroom by Risk Tier\n(Segment: Ratings >= {rating_threshold:.1f})", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("")
plt.ylabel("Typical Menu Price ($)", fontsize=12)
plt.ylim(target_median - 3, peer_median + 4) # Focuses the viewer on the gap
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend(loc='upper right', title="Capture Strategy")

plt.tight_layout()
plt.show()

# ======================================
# FINAL VALUE BRIDGE: PRICING HEADROOM 
# ======================================

# 1. Quality filter — use MEDIAN rating for robustness
rating_threshold = restaurant_summary_df["avg_review"].median()

pricing_headroom_df = (
    restaurant_value_flag
    .query("avg_review >= @rating_threshold")
    .assign(
        monetization_group=lambda x: np.where(
            x["under_monetized_flag"],
            "Under-monetized (High Quality)",
            "Priced Peers (High Quality)"
        )
    )
    .dropna(subset=["typical_menu_price"])
)

# 2. Plot
plt.figure(figsize=(9, 6))

palette = {
    "Under-monetized (High Quality)": "#AED6F1",
    "Priced Peers (High Quality)": "#D5DBDB"
}

ax = sns.boxplot(
    data=pricing_headroom_df,
    x="monetization_group",
    y="typical_menu_price",
    order=[
        "Under-monetized (High Quality)",
        "Priced Peers (High Quality)"
    ],
    showfliers=False,
    palette=palette,
    width=0.5
)

# 3. Median annotations
medians = (
    pricing_headroom_df
    .groupby("monetization_group")["typical_menu_price"]
    .median()
)

for i, label in enumerate(medians.index):
    ax.text(
        i,
        medians[label] + 0.25,
        f"${medians[label]:.2f}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

# 4. Explicit price gap annotation
price_gap = (
    medians["Priced Peers (High Quality)"] -
    medians["Under-monetized (High Quality)"]
)

ax.annotate(
    f"≈ ${price_gap:.2f} pricing headroom\nwithin similar quality",
    xy=(0.5, medians.mean()),
    xytext=(0.5, medians.max() + 2),
    arrowprops=dict(arrowstyle="->"),
    ha="center",
    fontsize=11
)

# 5. Labels
plt.title(
    f"Pricing Headroom Among High-Quality Restaurants\n(Ratings ≥ Portfolio Median: {rating_threshold:.1f})",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("")
plt.ylabel("Typical Menu Price ($)")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()



restaurant_value_flag["price_gap_vs_province"] = (
    restaurant_value_flag["province_median_price"]
    - restaurant_value_flag["typical_menu_price"]
)

pricing_gap_df = (
    restaurant_value_flag
    .query("avg_review >= @rating_threshold")
    .assign(
        group=lambda x: np.where(
            x["under_monetized_flag"],
            "Under-monetized (High Quality)",
            "Market-aligned Peers (High Quality)"
        )
    )
    .dropna(subset=["price_gap_vs_province"])
)

plt.figure(figsize=(9, 6))

ax = sns.boxplot(
    data=pricing_gap_df,
    x="group",
    y="price_gap_vs_province",
    showfliers=False,
    palette={
        "Under-monetized (High Quality)": "#AED6F1",
        "Market-aligned Peers (High Quality)": "#D5DBDB"
    }
)

plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.title(
    "Pricing Headroom Relative to Local Market Benchmarks\n(High-Quality Restaurants Only)",
    fontsize=14,
    fontweight="bold"
)
plt.ylabel("Price Gap vs Province Median ($)")
plt.xlabel("")
plt.tight_layout()
plt.show()

# ======================================
# FINAL VALUE BRIDGE: PRICING HEADROOM 
# ======================================

# 1. Prepare the Data
rating_threshold = restaurant_summary_df["avg_review"].median()

pricing_headroom_df = (
    restaurant_value_flag
    .query("avg_review >= @rating_threshold")
    .assign(
        monetization_group=lambda x: np.where(
            x["under_monetized_flag"],
            "Under-monetized (High Quality)",
            "Priced Peers (High Quality)"
        )
    )
    .dropna(subset=["typical_menu_price"])
)

# 2. Setup the Plot
plt.figure(figsize=(11, 7))
palette = {"Under-monetized (High Quality)": "#AED6F1", "Priced Peers (High Quality)": "#D5DBDB"}

ax = sns.boxplot(
    data=pricing_headroom_df,
    x="monetization_group",
    y="typical_menu_price",
    order=["Under-monetized (High Quality)", "Priced Peers (High Quality)"],
    showfliers=False,
    palette=palette,
    width=0.4
)

# 3. Calculate Medians & The Gap
medians = pricing_headroom_df.groupby("monetization_group")["typical_menu_price"].median()
u_med = medians["Under-monetized (High Quality)"]
p_med = medians["Priced Peers (High Quality)"]
total_gap = p_med - u_med

# 4. OVERLAY RISK TIERS (The 'Implementation' Logic)
# Tier 1: Low Risk (The first $1.50)
plt.axhspan(u_med, u_med + 1.5, color='green', alpha=0.1)
# Tier 2: Moderate Risk ($1.50 to $3.00)
plt.axhspan(u_med + 1.5, u_med + 3.0, color='orange', alpha=0.1)
# Tier 3: High Risk (Beyond $3.00 up to Peer Median)
plt.axhspan(u_med + 3.0, p_med, color='red', alpha=0.05)

# 5. Add Annotations
# Tier Labels
plt.text(-0.4, u_med + 0.6, "Tier 1: Low Risk\n(Inflationary)", color='green', fontweight='bold', fontsize=9)
plt.text(-0.4, u_med + 2.1, "Tier 2: Mod. Risk\n(Market Align)", color='darkorange', fontweight='bold', fontsize=9)
plt.text(-0.4, u_med + 3.6, "Tier 3: High Risk\n(Premium)", color='red', fontweight='bold', fontsize=9)

# Median Values
for i, val in enumerate([u_med, p_med]):
    ax.text(i, val + 0.3, f"${val:.2f}", ha="center", fontsize=12, fontweight="bold", color="#2C3E50")

# The Big Gap Arrow
ax.annotate('', xy=(0.5, u_med), xytext=(0.5, p_med),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
plt.text(0.55, (u_med + p_med)/2, f"Total Headroom: ${total_gap:.2f}", 
         va='center', fontweight='bold', fontsize=12, color='black')

# 6. Final Polish
plt.title(f"Commercial Upside: High-Quality Pricing Realignment\n(Filter: Ratings ≥ {rating_threshold:.1f})", 
          fontsize=15, fontweight='bold', pad=25)
plt.ylabel("Typical Menu Price ($)", fontsize=12)
plt.xlabel("")
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ======================================
# FINAL VALUE BRIDGE: PRICING HEADROOM (NORMALIZED)
# ======================================

restaurant_value_flag["price_gap_vs_province"] = (
    restaurant_value_flag["province_median_price"]
    - restaurant_value_flag["typical_menu_price"]
)

rating_threshold = restaurant_summary_df["avg_review"].median()

pricing_headroom_df = (
    restaurant_value_flag
    .query("avg_review >= @rating_threshold")
    .assign(
        monetization_group=lambda x: np.where(
            x["under_monetized_flag"],
            "Under-monetized (High Quality)",
            "Market-aligned Peers (High Quality)"
        )
    )
    .dropna(subset=["price_gap_vs_province"])
)

plt.figure(figsize=(11, 7))

palette = {
    "Under-monetized (High Quality)": "#AED6F1",
    "Market-aligned Peers (High Quality)": "#D5DBDB"
}

ax = sns.boxplot(
    data=pricing_headroom_df,
    x="monetization_group",
    y="price_gap_vs_province",
    order=[
        "Market-aligned Peers (High Quality)",
        "Under-monetized (High Quality)"
    ],
    showfliers=False,
    palette=palette,
    width=0.4
)

# Reference line: market parity
plt.axhline(0, color="black", linestyle="--", linewidth=1)

# Medians
medians = pricing_headroom_df.groupby("monetization_group")["price_gap_vs_province"].median()
u_med = medians["Under-monetized (High Quality)"]
p_med = medians["Market-aligned Peers (High Quality)"]
total_gap = u_med  # peers cluster around zero by definition

# Tier bands (NOW MEANINGFUL)
plt.axhspan(0, 1.5, color='green', alpha=0.1)
plt.axhspan(1.5, 3.0, color='orange', alpha=0.1)
plt.axhspan(3.0, max(u_med, 4), color='red', alpha=0.05)

# Tier labels
plt.text(-0.45, 0.6, "Tier 1: Low Risk\n(Inflationary)", color='green', fontweight='bold', fontsize=9)
plt.text(-0.45, 2.1, "Tier 2: Mod. Risk\n(Market Align)", color='darkorange', fontweight='bold', fontsize=9)
plt.text(-0.45, 3.6, "Tier 3: High Risk\n(Premium)", color='red', fontweight='bold', fontsize=9)

# Median annotation
ax.text(0, u_med + 0.1, f"+${u_med:.2f}", ha="center", fontsize=12, fontweight="bold")
ax.text(1, 0.05, "$0.00", ha="center", fontsize=12, fontweight="bold")

# Headroom arrow
ax.annotate(
    '',
    xy=(0.5, 0),
    xytext=(0.5, u_med),
    arrowprops=dict(arrowstyle='<->', lw=2)
)
plt.text(
    0.55,
    u_med / 2,
    f"Median pricing headroom: +${u_med:.2f}",
    va='center',
    fontweight='bold',
    fontsize=12
)

# Final formatting
plt.title(
    f"Commercial Upside: Pricing Headroom vs Local Market\n(High-Quality Restaurants, Ratings ≥ {rating_threshold:.1f})",
    fontsize=15,
    fontweight='bold',
    pad=25
)
plt.ylabel("Price Gap vs Province Median ($)")
plt.xlabel("")
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()



# =========================================
# SLIDE 25 — PRICING RISK DECISION GRID
# Where to Scale vs Pilot
# (CONTINUES FROM EXISTING PIPELINE)
# =========================================

# ---------------------------------------------------------
# 1. Start from existing restaurant-level dataframe
# ---------------------------------------------------------
province_grid = (
    restaurant_summary_df
    .dropna(subset=["province_clean", "typical_menu_price", "avg_review"])
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        rating_mean=("avg_review", "mean"),
        rating_std=("avg_review", "std"),          # Execution volatility
        price_std=("typical_menu_price", "std"),  # Pricing dispersion
        price_median=("typical_menu_price", "median")
    )
    .reset_index()
)

# Keep statistically meaningful provinces only
province_grid = province_grid[
    province_grid["restaurant_count"] >= 10
].copy()

# ---------------------------------------------------------
# 2. Define decision thresholds (robust medians)
# ---------------------------------------------------------
price_dispersion_cut = province_grid["price_std"].median()
rating_volatility_cut = province_grid["rating_std"].median()

# ---------------------------------------------------------
# 3. Classify Market DNA (explicit, reusable)
# ---------------------------------------------------------
province_grid["market_dna"] = np.select(
    [
        (province_grid["price_std"] <= price_dispersion_cut) &
        (province_grid["rating_std"] <= rating_volatility_cut),

        (province_grid["price_std"] > price_dispersion_cut) &
        (province_grid["rating_std"] <= rating_volatility_cut),

        (province_grid["rating_std"] > rating_volatility_cut)
    ],
    [
        "Stable / Scalable",
        "Chaotic Pricing (Pilot Only)",
        "Execution Risk"
    ],
    default="Other"
)

# ---------------------------------------------------------
# 4. Plot the Decision Grid
# ---------------------------------------------------------
plt.figure(figsize=(11, 7))

sns.scatterplot(
    data=province_grid,
    x="price_std",
    y="rating_std",
    size="restaurant_count",
    hue="market_dna",
    sizes=(250, 1400),
    alpha=0.8,
    edgecolor="black"
)

# Decision boundaries
plt.axvline(price_dispersion_cut, linestyle="--", color="black", linewidth=1)
plt.axhline(rating_volatility_cut, linestyle="--", color="black", linewidth=1)

# Reverse Y-axis so lower volatility (more stable execution) is at the top
plt.gca().invert_yaxis()

# Province labels
for _, row in province_grid.iterrows():
    plt.text(
        row["price_std"] + 0.05,
        row["rating_std"],
        row["province_clean"],
        fontsize=9
    )

plt.title(
    "Pricing Risk Decision Grid: Where to Scale vs Pilot",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Price Dispersion (Std Dev of Typical Menu Price)")
plt.ylabel("Rating Volatility (Std Dev of Avg Review)")
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================
# SLIDE 25 — PRICING RISK DIAGNOSTIC GRID
# (No Color Grouping)
# =========================================

# ---------------------------------------------------------
# 1. Province-level aggregation (from existing pipeline)
# ---------------------------------------------------------
province_grid = (
    restaurant_summary_df
    .dropna(subset=["province_clean", "typical_menu_price", "avg_review"])
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        rating_mean=("avg_review", "mean"),
        rating_std=("avg_review", "std"),          # Execution volatility
        price_std=("typical_menu_price", "std"),  # Pricing dispersion
        price_median=("typical_menu_price", "median")
    )
    .reset_index()
)

# Keep statistically meaningful provinces only
province_grid = province_grid[
    province_grid["restaurant_count"] >= 10
].copy()

# ---------------------------------------------------------
# 2. Define reference lines (robust medians)
# ---------------------------------------------------------
price_dispersion_cut = province_grid["price_std"].median()
rating_volatility_cut = province_grid["rating_std"].median()

# ---------------------------------------------------------
# 3. Plot — continuous diagnostic grid
# ---------------------------------------------------------
plt.figure(figsize=(11, 7))

sns.scatterplot(
    data=province_grid,
    x="price_std",
    y="rating_std",
    size="restaurant_count",
    sizes=(250, 1400),
    color="#5b6b7a",        # Single neutral color
    alpha=0.8,
    edgecolor="black"
)

# Decision reference lines
plt.axvline(price_dispersion_cut, linestyle="--", color="black", linewidth=1)
plt.axhline(rating_volatility_cut, linestyle="--", color="black", linewidth=1)

# Reverse Y-axis: lower volatility (more stable execution) at the top
plt.gca().invert_yaxis()

# Province labels
for _, row in province_grid.iterrows():
    plt.text(
        row["price_std"] + 0.05,
        row["rating_std"],
        row["province_clean"],
        fontsize=9,
        alpha=0.9
    )

# ---------------------------------------------------------
# 4. Quadrant descriptors (descriptive, not prescriptive)
# ---------------------------------------------------------
plt.text(
    price_dispersion_cut * 0.35,
    rating_volatility_cut * 0.85,
    "Tight pricing\nStable execution",
    fontsize=11,
    fontweight="bold"
)

plt.text(
    price_dispersion_cut * 1.05,
    rating_volatility_cut * 0.85,
    "Wide pricing\nStable execution",
    fontsize=11,
    fontweight="bold"
)

plt.text(
    price_dispersion_cut * 0.35,
    rating_volatility_cut * 1.15,
    "Tight pricing\nVolatile execution",
    fontsize=11,
    fontweight="bold"
)

plt.text(
    price_dispersion_cut * 1.05,
    rating_volatility_cut * 1.15,
    "Wide pricing\nVolatile execution",
    fontsize=11,
    fontweight="bold"
)

# ---------------------------------------------------------
# 5. Final formatting
# ---------------------------------------------------------
plt.title(
    "Pricing Risk Diagnostic: Market Structure vs Execution Stability",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Price Dispersion (Std Dev of Typical Menu Price)",
    fontsize=12
)

plt.ylabel(
    "Rating Volatility (Std Dev of Avg Review)",
    fontsize=12
)

plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# SLIDE 27 — MONETIZABLE PRICING HEADROOM
# Applying Market DNA to Restaurant-Level Price Gaps
# =========================================================

# ---------------------------------------------------------
# 5. Bring Market DNA back to restaurant level
# ---------------------------------------------------------
slide27_df = restaurant_summary_df.merge(
    province_grid[["province_clean", "market_dna", "price_median"]],
    on="province_clean",
    how="inner"   # inner = only provinces we already validated (>=20 stores)
)

# ---------------------------------------------------------
# 6. Compute price gap vs local market (COL-normalized)
# ---------------------------------------------------------
slide27_df["price_gap_pct"] = (
    (slide27_df["price_median"] - slide27_df["typical_menu_price"])
    / slide27_df["price_median"]
) * 100

# Optional: cap extreme tails for presentation clarity (NOT analysis filtering)
# slide27_df = slide27_df[
#     slide27_df["price_gap_pct"].between(-40, 20)
# ].copy()

# ---------------------------------------------------------
# 7. Boxplot — How much headroom exists by Market DNA?
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.set_style("white")

sns.boxplot(
    data=slide27_df,
    x="market_dna",
    y="price_gap_pct",
    showfliers=True,
    order=[
        "Stable / Scalable",
        "Chaotic Pricing (Pilot Only)",
        "Execution Risk"
    ],
    palette={
        "Stable / Scalable": "#2ecc71",
        "Chaotic Pricing (Pilot Only)": "#f39c12",
        "Execution Risk": "#e74c3c"
    }
)

# Local parity reference
plt.axhline(0, linestyle="--", color="black", linewidth=1)

# ---------------------------------------------------------
# 8. Formatting (IC-ready)
# ---------------------------------------------------------
plt.title(
    "Monetizable Pricing Headroom by Market Execution DNA",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel("Price Gap vs Local Province Median (%)")
plt.xlabel("")

plt.tight_layout()
plt.show()




# =========================================
# SLIDE 27 — POST-DEAL PLAYBOOK
# Where Pricing Upside Exists Without Breaking Execution
# (CONTINUES FROM SLIDE 25 PIPELINE)
# =========================================

# ---------------------------------------------------------
# 1. Attach Market DNA back to restaurant-level data
# ---------------------------------------------------------
slide27_df = restaurant_summary_df.merge(
    province_grid[["province_clean", "market_dna"]],
    on="province_clean",
    how="inner"
)

# ---------------------------------------------------------
# 2. Province-normalized price gap (cost-of-living safe)
# ---------------------------------------------------------
slide27_df["province_median_price"] = (
    slide27_df
    .groupby("province_clean")["typical_menu_price"]
    .transform("median")
)

slide27_df["price_gap_pct"] = (
    (slide27_df["province_median_price"] - slide27_df["typical_menu_price"])
    / slide27_df["province_median_price"]
) * 100

# Optional: cap extreme tails for presentation clarity (NOT analysis filtering)
slide27_df = slide27_df[
    slide27_df["price_gap_pct"].between(-40, 20)
].copy()


# ---------------------------------------------------------
# 3. Explicit tier order (critical for narrative control)
# ---------------------------------------------------------
tier_order = [
    "Stable / Scalable",
    "Chaotic Pricing (Pilot Only)",
    "Execution Risk"
]

palette = {
    "Stable / Scalable": "#2ecc71",
    "Chaotic Pricing (Pilot Only)": "#f1c40f",
    "Execution Risk": "#e74c3c"
}

# ---------------------------------------------------------
# 4. Plot boxplot + jitter (distribution focus)
# ---------------------------------------------------------
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

sns.boxplot(
    data=slide27_df,
    x="market_dna",
    y="price_gap_pct",
    order=tier_order,
    palette=palette,
    width=0.5,
    fliersize=0
)

sns.stripplot(
    data=slide27_df,
    x="market_dna",
    y="price_gap_pct",
    order=tier_order,
    color="black",
    alpha=0.15,
    size=4,
    jitter=True
)

# Zero reference line
plt.axhline(0, linestyle="--", color="black", linewidth=1)

# ---------------------------------------------------------
# 5. Annotate N-count, Avg Rating, and P75 (key insight)
# ---------------------------------------------------------
for i, tier in enumerate(tier_order):
    tier_data = slide27_df[slide27_df["market_dna"] == tier]

    n_count = tier_data.shape[0]
    avg_rating = tier_data["avg_review"].mean()
    p75 = tier_data["price_gap_pct"].quantile(0.75)

    # N-count + Avg Rating (top)
    plt.text(
        i,
        slide27_df["price_gap_pct"].max() + 4,
        f"n = {n_count}\nAvg Rating: {avg_rating:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

    # P75 label (this is the monetizable signal)
    plt.text(
        i,
        p75,
        f"P75: {p75:.0f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

# ---------------------------------------------------------
# 6. Final formatting
# ---------------------------------------------------------
plt.title(
    "Post-Deal Playbook: Where Pricing Upside Exists Without Breaking Execution",
    fontsize=16,
    fontweight="bold",
    pad=20
)

plt.xlabel("")
plt.ylabel(
    "Price Headroom vs Local Province Median (%)\n(Positive = Under-monetized)",
    fontsize=12
)

plt.tight_layout()
plt.show()





# =========================================
# SLIDE 25 — PRICING RISK DECISION GRID
# Where to Scale vs Pilot
# =========================================

# ---------------------------------------------------------
# 1. Province-level aggregation
# ---------------------------------------------------------
province_grid = (
    restaurant_summary_df
    .dropna(subset=["province_clean", "typical_menu_price", "avg_review"])
    .groupby("province_clean")
    .agg(
        restaurant_count=("id", "nunique"),
        rating_std=("avg_review", "std"),           # Execution volatility
        price_std=("typical_menu_price", "std")    # Pricing dispersion
    )
    .reset_index()
)

# Optional: minimum sample size for stability
province_grid = province_grid[
    province_grid["restaurant_count"] >= 10
].copy()

# ---------------------------------------------------------
# 2. Robust decision thresholds (MEDIANS — allowed & correct)
# ---------------------------------------------------------
price_dispersion_cut = province_grid["price_std"].median()
rating_volatility_cut = province_grid["rating_std"].median()

# ---------------------------------------------------------
# 3. Market DNA classification (3 groups, decision-ready)
# ---------------------------------------------------------
province_grid["market_dna"] = np.select(
    [
        (province_grid["price_std"] <= price_dispersion_cut) &
        (province_grid["rating_std"] <= rating_volatility_cut),

        (province_grid["price_std"] > price_dispersion_cut) &
        (province_grid["rating_std"] <= rating_volatility_cut),

        (province_grid["rating_std"] > rating_volatility_cut)
    ],
    [
        "Stable / Scalable",
        "Chaotic Pricing (Pilot Only)",
        "Execution Risk"
    ],
    default="Other"
)

# ---------------------------------------------------------
# 4. FIXED order + FIXED color semantics (THIS SOLVES YOUR ISSUE)
# ---------------------------------------------------------
market_order = [
    "Stable / Scalable",
    "Chaotic Pricing (Pilot Only)",
    "Execution Risk"
]

palette = {
    "Stable / Scalable": "#2ecc71",               # GREEN
    "Chaotic Pricing (Pilot Only)": "#f1c40f",    # AMBER
    "Execution Risk": "#e74c3c"                   # RED
}

# ---------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------
plt.figure(figsize=(11, 7))
sns.set_theme(style="whitegrid")

sns.scatterplot(
    data=province_grid,
    x="price_std",
    y="rating_std",
    size="restaurant_count",
    hue="market_dna",
    hue_order=market_order,
    palette=palette,
    sizes=(250, 1400),
    alpha=0.85,
    edgecolor="black"
)

# Decision boundaries
plt.axvline(price_dispersion_cut, linestyle="--", color="black", linewidth=1)
plt.axhline(rating_volatility_cut, linestyle="--", color="black", linewidth=1)

# Reverse Y-axis (lower volatility = better execution at top)
plt.gca().invert_yaxis()

# Province labels
for _, row in province_grid.iterrows():
    plt.text(
        row["price_std"] + 0.05,
        row["rating_std"],
        row["province_clean"],
        fontsize=9
    )

# ---------------------------------------------------------
# 6. Titles & labels
# ---------------------------------------------------------
plt.title(
    "Pricing Risk Decision Grid: Where to Scale vs Pilot",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Price Dispersion (Std Dev of Typical Menu Price)")
plt.ylabel("Rating Volatility (Std Dev of Avg Review)")

plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================
# SLIDE 27 — POST-DEAL PLAYBOOK
# Monetizable Pricing Upside by Risk Tier
# =========================================

# ---------------------------------------------------------
# 1. Assign Market DNA from Slide 25 logic
# ---------------------------------------------------------
def classify_market(row):
    if row["rating_std"] <= rating_volatility_cut and row["price_std"] <= price_dispersion_cut:
        return "Stable / Scalable"
    elif row["rating_std"] <= rating_volatility_cut and row["price_std"] > price_dispersion_cut:
        return "Chaotic Pricing (Pilot Only)"
    else:
        return "Execution Risk"

province_grid["market_dna"] = province_grid.apply(classify_market, axis=1)

# ---------------------------------------------------------
# 2. Attach Market DNA to restaurant-level data
# ---------------------------------------------------------
slide27_df = restaurant_summary_df.merge(
    province_grid[["province_clean", "market_dna"]],
    on="province_clean",
    how="inner"
)

# ---------------------------------------------------------
# 3. Province-normalized price gap (COL safe)
# ---------------------------------------------------------
slide27_df["province_median_price"] = (
    slide27_df
    .groupby("province_clean")["typical_menu_price"]
    .transform("median")
)

slide27_df["price_gap_pct"] = (
    (slide27_df["province_median_price"] - slide27_df["typical_menu_price"])
    / slide27_df["province_median_price"]
) * 100

# ---------------------------------------------------------
# 4. Plot distribution (NO tail capping)
# ---------------------------------------------------------
tier_order = [
    "Stable / Scalable",
    "Chaotic Pricing (Pilot Only)",
    "Execution Risk"
]

palette = {
    "Stable / Scalable": "#2ecc71",
    "Chaotic Pricing (Pilot Only)": "#f1c40f",
    "Execution Risk": "#e74c3c"
}

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

sns.boxplot(
    data=slide27_df,
    x="market_dna",
    y="price_gap_pct",
    order=tier_order,
    palette=palette,
    width=0.5,
    fliersize=0
)

sns.stripplot(
    data=slide27_df,
    x="market_dna",
    y="price_gap_pct",
    order=tier_order,
    color="black",
    alpha=0.15,
    size=4,
    jitter=True
)

plt.axhline(0, linestyle="--", color="black", linewidth=1)

# ---------------------------------------------------------
# 5. Annotate key decision statistics
# ---------------------------------------------------------
for i, tier in enumerate(tier_order):
    tier_data = slide27_df[slide27_df["market_dna"] == tier]

    n_count = tier_data.shape[0]
    avg_rating = tier_data["avg_review"].mean()
    p75 = tier_data["price_gap_pct"].quantile(0.75)

    plt.text(
        i,
        slide27_df["price_gap_pct"].max() * 0.95,
        f"n = {n_count}\nAvg Rating: {avg_rating:.2f}",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold"
    )

    plt.text(
        i,
        p75,
        f"P75: {p75:.0f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.title(
    "Post-Deal Playbook: Monetizable Pricing Upside vs Earnings Risk",
    fontsize=16,
    fontweight="bold",
    pad=20
)

plt.xlabel("")
plt.ylabel(
    "Price Headroom vs Local Province Median (%)\n(Positive = Under-monetized)",
    fontsize=12
)

plt.tight_layout()
plt.show()


# =========================================
# SLIDE 27 — POST-DEAL PLAYBOOK
# Monetizable Pricing Upside by Risk Tier
# =========================================

# ---------------------------------------------------------
# 1. Attach Market DNA (from Slide 25) to restaurants
# ---------------------------------------------------------
slide27_df = restaurant_summary_df.merge(
    province_grid[["province_clean", "market_dna"]],
    on="province_clean",
    how="inner"
)

# ---------------------------------------------------------
# 2. Province-normalized price gap
# (controls for cost-of-living differences)
# ---------------------------------------------------------
slide27_df["province_median_price"] = (
    slide27_df
    .groupby("province_clean")["typical_menu_price"]
    .transform("median")
)

slide27_df["price_gap_pct"] = (
    (slide27_df["province_median_price"] - slide27_df["typical_menu_price"])
    / slide27_df["province_median_price"]
) * 100

# ---------------------------------------------------------
# 3. Plot setup
# ---------------------------------------------------------
tier_order = [
    "Stable / Scalable",
    "Chaotic Pricing (Pilot Only)",
    "Execution Risk"
]

palette = {
    "Stable / Scalable": "#2ecc71",
    "Chaotic Pricing (Pilot Only)": "#f1c40f",
    "Execution Risk": "#e74c3c"
}

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# 4. Boxplot (distribution view)
# ---------------------------------------------------------
sns.boxplot(
    data=slide27_df,
    x="market_dna",
    y="price_gap_pct",
    order=tier_order,
    palette=palette,
    width=0.5,
    fliersize=0
)

# Light jitter for density
sns.stripplot(
    data=slide27_df,
    x="market_dna",
    y="price_gap_pct",
    order=tier_order,
    color="black",
    alpha=0.10,
    size=3,
    jitter=True
)

# Zero reference line
plt.axhline(0, linestyle="--", color="black", linewidth=1)

# ---------------------------------------------------------
# 5. Focus view (DOES NOT remove data)
# ---------------------------------------------------------
plt.ylim(-60, 60)

# ---------------------------------------------------------
# 6. Annotate key metrics (simple & decision-focused)
# ---------------------------------------------------------
y_top = plt.gca().get_ylim()[1]

for i, tier in enumerate(tier_order):
    tier_data = slide27_df[slide27_df["market_dna"] == tier]

    n_count = tier_data.shape[0]
    avg_rating = tier_data["avg_review"].mean()
    p75 = tier_data["price_gap_pct"].quantile(0.75)

    # Sample size + quality
    plt.text(
        i,
        y_top * 0.95,
        f"n = {n_count}\nAvg Rating: {avg_rating:.2f}",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold"
    )

    # Monetizable upside signal
    plt.text(
        i,
        p75,
        f"P75: {p75:.0f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

# ---------------------------------------------------------
# 7. Titles & labels
# ---------------------------------------------------------
plt.title(
    "Post-Deal Playbook: Monetizable Pricing Upside vs Earnings Risk",
    fontsize=16,
    fontweight="bold",
    pad=20
)

plt.xlabel("")
plt.ylabel(
    "Price Headroom vs Local Province Median (%)\n(Positive = Under-monetized)",
    fontsize=12
)

plt.tight_layout()
plt.show()
