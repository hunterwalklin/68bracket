"""Team name normalization across different data sources."""

import re

# Mapping of common alternate names/abbreviations to canonical Sports-Reference school_id
# This handles discrepancies between poll data, bracket data, and stats pages
NAME_ALIASES = {
    # Common abbreviations
    "UConn": "connecticut",
    "Connecticut": "connecticut",
    "UCONN": "connecticut",
    "UNC": "north-carolina",
    "North Carolina": "north-carolina",
    "UCLA": "ucla",
    "USC": "southern-california",
    "SMU": "southern-methodist",
    "Southern Methodist": "southern-methodist",
    "LSU": "louisiana-state",
    "Louisiana State": "louisiana-state",
    "TCU": "texas-christian",
    "Texas Christian": "texas-christian",
    "UCF": "central-florida",
    "Central Florida": "central-florida",
    "UNLV": "nevada-las-vegas",
    "Nevada-Las Vegas": "nevada-las-vegas",
    "VCU": "virginia-commonwealth",
    "Virginia Commonwealth": "virginia-commonwealth",
    "BYU": "brigham-young",
    "Brigham Young": "brigham-young",
    "Ole Miss": "mississippi",
    "Mississippi": "mississippi",
    "Pitt": "pittsburgh",
    "Pittsburgh": "pittsburgh",
    "Miami (FL)": "miami-fl",
    "Miami FL": "miami-fl",
    "Miami (OH)": "miami-oh",
    "Miami OH": "miami-oh",
    "Saint Mary's": "saint-marys-ca",
    "Saint Mary's (CA)": "saint-marys-ca",
    "St. Mary's": "saint-marys-ca",
    "St. Mary's (CA)": "saint-marys-ca",
    "St. John's": "st-johns-ny",
    "St. John's (NY)": "st-johns-ny",
    "Saint John's": "st-johns-ny",
    "St. Bonaventure": "st-bonaventure",
    "Saint Bonaventure": "st-bonaventure",
    "St. Peter's": "saint-peters",
    "Saint Peter's": "saint-peters",
    "NC State": "north-carolina-state",
    "North Carolina State": "north-carolina-state",
    "Penn State": "penn-state",
    "Penn St.": "penn-state",
    "Ohio State": "ohio-state",
    "Ohio St.": "ohio-state",
    "Michigan State": "michigan-state",
    "Michigan St.": "michigan-state",
    "Mississippi State": "mississippi-state",
    "Miss. State": "mississippi-state",
    "Iowa State": "iowa-state",
    "Kansas State": "kansas-state",
    "Colorado State": "colorado-state",
    "Boise State": "boise-state",
    "San Diego State": "san-diego-state",
    "Fresno State": "fresno-state",
    "Oregon State": "oregon-state",
    "Washington State": "washington-state",
    "Wichita State": "wichita-state",
    "Utah State": "utah-state",
    "Arizona State": "arizona-state",
    "Florida State": "florida-state",
    "Fla. State": "florida-state",
    "Murray State": "murray-state",
    "Kent State": "kent-state",
    "Baylor": "baylor",
    "Gonzaga": "gonzaga",
    "Villanova": "villanova",
    "Duke": "duke",
    "Kentucky": "kentucky",
    "Kansas": "kansas",
    "Virginia": "virginia",
    "Purdue": "purdue",
    "Houston": "houston",
    "Alabama": "alabama",
    "Tennessee": "tennessee",
    "Marquette": "marquette",
    "Creighton": "creighton",
    "FGCU": "florida-gulf-coast",
    "Florida Gulf Coast": "florida-gulf-coast",
    "Loyola Chicago": "loyola-il",
    "Loyola (IL)": "loyola-il",
    "Loyola-Chicago": "loyola-il",
    "UNC Asheville": "north-carolina-asheville",
    "ETSU": "east-tennessee-state",
    "East Tennessee State": "east-tennessee-state",
    "MTSU": "middle-tennessee",
    "Middle Tennessee": "middle-tennessee",
    "UNC Wilmington": "north-carolina-wilmington",
    "UNC Greensboro": "north-carolina-greensboro",
    "SIU Edwardsville": "southern-illinois-edwardsville",
    "SIUE": "southern-illinois-edwardsville",
    "LIU": "long-island-university",
    "Long Island": "long-island-university",
    "FDU": "fairleigh-dickinson",
    "Fairleigh Dickinson": "fairleigh-dickinson",
    "A&M": "texas-am",
    "Texas A&M": "texas-am",
    # WarrenNolan-specific name variants
    "Saint Mary's College": "saint-marys-ca",
    "UC San Diego": "california-san-diego",
    "UC Irvine": "california-irvine",
    "UC Davis": "california-davis",
    "UC Riverside": "california-riverside",
    "UC Santa Barbara": "california-santa-barbara",
    "McNeese": "mcneese-state",
    "Mount Saint Mary's": "mount-st-marys",
    "Saint Francis (PA)": "saint-francis-pa",
    "Saint Joseph's": "saint-josephs",
    "UAB": "alabama-birmingham",
    "UMBC": "maryland-baltimore-county",
    "UMKC": "missouri-kansas-city",
    "UIC": "illinois-chicago",
    "UNI": "northern-iowa",
    "NIU": "northern-illinois",
    "SFA": "stephen-f-austin",
    "UTEP": "texas-el-paso",
    "UTSA": "texas-san-antonio",
    "App State": "appalachian-state",
    "Southern Miss": "southern-mississippi",
    "Grambling": "grambling-state",
    "Southeast Missouri": "southeast-missouri-state",
    "Purdue Fort Wayne": "purdue-fort-wayne",
    "Le Moyne": "le-moyne",
    "Queens University": "queens-nc",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name to lowercase-hyphenated school_id format.

    Handles common abbreviations and formatting differences.
    """
    if not name or not isinstance(name, str):
        return ""

    name = name.strip()

    # Remove tournament markers
    name = re.sub(r"\s*NCAA$", "", name)
    name = re.sub(r"\s*\(\d+\)$", "", name)  # Remove seed markers like "(1)"
    name = name.strip()

    # Check alias map first
    if name in NAME_ALIASES:
        return NAME_ALIASES[name]

    # Try case-insensitive alias lookup
    name_lower = name.lower()
    for alias, school_id in NAME_ALIASES.items():
        if alias.lower() == name_lower:
            return school_id

    # Default: convert to SR school_id format
    # Lowercase, replace spaces/special chars with hyphens
    school_id = name.lower()
    school_id = re.sub(r"[.'()]", "", school_id)
    school_id = re.sub(r"[&]", "and", school_id)
    school_id = re.sub(r"\s+", "-", school_id)
    school_id = re.sub(r"-+", "-", school_id)
    school_id = school_id.strip("-")

    return school_id


def merge_on_team(left: "pd.DataFrame", right: "pd.DataFrame",
                  left_team_col: str = "team", right_team_col: str = "team",
                  season_col: str = "season", how: str = "left") -> "pd.DataFrame":
    """Merge two DataFrames on normalized team names and season.

    Creates temporary normalized columns for matching, then drops them.
    """
    import pandas as pd

    left = left.copy()
    right = right.copy()

    # Prefer school_id if available in both
    if "school_id" in left.columns and "school_id" in right.columns:
        merge_cols = ["school_id", season_col]
        right_merge = right[[c for c in right.columns if c not in left.columns or c in merge_cols]]
        return left.merge(right_merge, on=merge_cols, how=how)

    # Normalize team names for merging
    left["_merge_key"] = left[left_team_col].apply(normalize_team_name)
    right["_merge_key"] = right[right_team_col].apply(normalize_team_name)

    merge_cols = ["_merge_key", season_col]
    right_merge = right[[c for c in right.columns if c not in left.columns or c in merge_cols]]

    result = left.merge(right_merge, on=merge_cols, how=how)
    result = result.drop(columns=["_merge_key"], errors="ignore")

    return result
