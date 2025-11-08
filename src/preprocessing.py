import itertools
import os.path
import numpy as np
import pandas as pd


def expand_grid(data_dict):
    """
    Create a dataframe from all combinations of provided lists or arrays.

    This function takes a dictionary of lists or arrays and computes the cartesian product of these lists or arrays.
    Each unique combination of elements will form a row in the resulting dataframe. The keys of the dictionary will
    be used as column names in the dataframe.

    Parameters:
    data_dict (dict): A dictionary where keys are column names and values are lists or arrays containing data.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the cartesian product of the provided lists or arrays.

    Example:
        data_dict = {'height': [60, 70], 'weight': [100, 150, 200]}

        expand_grid(data_dict)

       height  weight
    0      60     100
    1      60     150
    2      60     200
    3      70     100
    4      70     150
    5      70     200
    """
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def load_data_labeled_anon_and_transform_to_grid(path_to_data,
                                                 print_to_excel=False):

    data_labeled_anon = pd.read_excel(os.path.join(path_to_data, "input-data/data_labeled_anon.xlsx"),
                                      sheet_name='Data')

    # drop row because we do not have this file
    data_labeled_anon = data_labeled_anon[data_labeled_anon["ReportName"] != "deutschebank_2011_en_pdf"]
    data_labeled_anon.loc[:, "ReportName"].drop_duplicates()[-5:]

    data_dict = {'ReportName': data_labeled_anon.loc[:, "ReportName"].drop_duplicates(),
                 'scope': range(1, 4),
                 'year': range(2010, 2026)}
    res = expand_grid(data_dict)

    res = pd.merge(res, data_labeled_anon, how="left", left_on=["ReportName", "scope", "year"],
                   right_on=["ReportName", "Scope", "Year"])

    res = res.rename(
        columns={"scope": "scope_man", "year": "year_man",
                 "Value": "value_man", "Unit": "unit_man",
                 "Page": "page_man",
                 "Value Name": "val_name_man", "Type": "type_man"})
    res = res[['ReportName', 'scope_man', 'year_man', 'value_man',
               'unit_man', 'page_man', 'val_name_man', 'type_man']]


    # replace wrong values with correct ones
    res.loc[(res["ReportName"] == "continental_2013_en.pdf") & (res["scope_man"] == 1) & (res["year_man"] == 2013), "value_man"] = 659
    res.loc[(res["ReportName"] == "puma_2013_en.pdf") & (res["scope_man"] == 2) & (res["year_man"] == 2013), "value_man"] = 27835
    res.loc[(res["ReportName"] == "deutschepost_2012_en.pdf") & (res["page_man"] == 109), "value_man"] = np.nan
    res.loc[(res["ReportName"] == "deutschepost_2012_en.pdf") & (res["page_man"] == 109), "unit_man"] = np.nan
    res.loc[(res["ReportName"] == "deutschepost_2012_en.pdf") & (res["page_man"] == 109), "val_name_man"] = np.nan
    res.loc[(res["ReportName"] == "deutschepost_2012_en.pdf") & (res["page_man"] == 109), "type_man"] = np.nan
    res.loc[(res["ReportName"] == "deutschepost_2012_en.pdf") & (res["page_man"] == 109), "page_man"] = np.nan
    res.loc[(res["ReportName"] == "novonordisk_2020_en.pdf") & ((res["page_man"] == 278) | (res["page_man"] == 306)), "value_man"] = np.nan
    res.loc[(res["ReportName"] == "novonordisk_2020_en.pdf") & ((res["page_man"] == 278) | (res["page_man"] == 306)), "unit_man"] = np.nan
    res.loc[(res["ReportName"] == "novonordisk_2020_en.pdf") & ((res["page_man"] == 278) | (res["page_man"] == 306)), "val_name_man"] = np.nan
    res.loc[(res["ReportName"] == "novonordisk_2020_en.pdf") & ((res["page_man"] == 278) | (res["page_man"] == 306)), "type_man"] = np.nan
    res.loc[(res["ReportName"] == "novonordisk_2020_en.pdf") & ((res["page_man"] == 278) | (res["page_man"] == 306)), "page_man"] = np.nan
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & ((res["year_man"] == 2015) | (res["year_man"] == 2016) | (res["year_man"] == 2017)), "unit_man"] = "t"
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & ((res["year_man"] == 2015) | (res["year_man"] == 2016) | (res["year_man"] == 2017)), "page_man"] = 16
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & ((res["year_man"] == 2015) | (res["year_man"] == 2016) | (res["year_man"] == 2017)), "type_man"] = "Table"
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & ((res["year_man"] == 2015) | (res["year_man"] == 2016) | (res["year_man"] == 2017)) & (res["scope_man"] == 1), "val_name_man"] = "Scope 1 - Direct CO2e emissions fossil fuels (T)"
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & ((res["year_man"] == 2015) | (res["year_man"] == 2016) | (res["year_man"] == 2017)) & (res["scope_man"] == 2), "val_name_man"] = "Scope 2 - Indirect CO 2 e emissions electricity & steam (T)"
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & ((res["year_man"] == 2015) | (res["year_man"] == 2016) | (res["year_man"] == 2017)) & (res["scope_man"] == 3), "val_name_man"] = "Scope 3 - Other indirect CO2e emissions [T]"
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & (res["year_man"] == 2015) & (res["scope_man"] == 1), "value_man"] = 7296
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & (res["year_man"] == 2015) & (res["scope_man"] == 2), "value_man"] = 35591
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & (res["year_man"] == 2015) & (res["scope_man"] == 3), "value_man"] = 192305
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & (res["year_man"] == 2016) & (res["scope_man"] == 1), "value_man"] = 6854
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & (res["year_man"] == 2016) & (res["scope_man"] == 2), "value_man"] = 37300
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & (res["year_man"] == 2016) & (res["scope_man"] == 3), "value_man"] = 196896
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & (res["year_man"] == 2017) & (res["scope_man"] == 1), "value_man"] = 7678
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & (res["year_man"] == 2017) & (res["scope_man"] == 2), "value_man"] = 40029
    res.loc[(res["ReportName"] == "puma_2018_en.pdf") & (res["year_man"] == 2017) & (res["scope_man"] == 3), "value_man"] = 208525
    res.loc[(res["ReportName"] == "volkswagen_2019_en.pdf") & ((res["year_man"] == 2018) | (res["year_man"] == 2019)) & (res["scope_man"] == 3), "unit_man"] = "t CO2"
    res.loc[(res["ReportName"] == "volkswagen_2019_en.pdf") & ((res["year_man"] == 2018) | (res["year_man"] == 2019)) & (res["scope_man"] == 3), "page_man"] = 71
    res.loc[(res["ReportName"] == "volkswagen_2019_en.pdf") & ((res["year_man"] == 2018) | (res["year_man"] == 2019)) & (res["scope_man"] == 3), "val_name_man"] = "Total of reported Scope 3 GHG emissions (cars and light commercial vehicles)"
    res.loc[(res["ReportName"] == "volkswagen_2019_en.pdf") & ((res["year_man"] == 2018) | (res["year_man"] == 2019)) & (res["scope_man"] == 3), "type_man"] = "Table"
    res.loc[(res["ReportName"] == "volkswagen_2019_en.pdf") & (res["year_man"] == 2018) & (res["scope_man"] == 3), "value_man"] = 427529210
    res.loc[(res["ReportName"] == "volkswagen_2019_en.pdf") & (res["year_man"] == 2019) & (res["scope_man"] == 3), "value_man"] = 447420740

    # add comments
    res['ms_comment_man'] = ""  # initiate string column
    res.loc[(res["ReportName"] == "deutschebank_2017_en.pdf"), "ms_comment_man"] = "Historic values are on page 76, forecasted values are on page 86"
    res.loc[(res["ReportName"] == "deutschepost_2012_en.pdf"), "ms_comment_man"] = "I deleted forecasted values that the human annotator found in some external document."
    res.loc[(res["ReportName"] == "novonordisk_2020_en.pdf"), "ms_comment_man"] = "I deleted 2018/2019 values (scope 2) that the human annotator found in some external document."
    res.loc[(res["ReportName"] == "walmart_2017_en.pdf"), "ms_comment_man"] = "p. 56 has a chart with Carbon emissions (Scope 1 and 2), but no distinction between them"
    res.loc[(res["ReportName"] == "apple_2021_en.pdf"), "ms_comment_man"] = "p. 67 enthält die meisten Infos. Die Seiten 12,14,75,89 enthalten Zahlen für weitere Jahre und ggf. mit teils abweichenden Angaben"
    res.loc[(res["ReportName"] == "mercedesbenzgroup_2021_en.pdf"), "ms_comment_man"] = "Challenges: Mercedes reports on p. 153 Scope 1, Scope 2-market-based, Scope 2-location-based, Total-market-based, and Total-location based. From page 140 one can CALCULATE values for scope 3, separately for cars&vans"
    res.loc[(res["ReportName"] == "pfizer_2019_en.pdf"), "ms_comment_man"] = "Challenge: Numbers from chart cannot be copied"
    res.loc[(res["ReportName"] == "chevron_2020_en.pdf"), "ms_comment_man"] = "Challenge: a table that runs across three pages"
    res.loc[(res["ReportName"] == "samsung_2018_en.pdf"), "ms_comment_man"] = "Related charts are on p. 047 and 059, which makes the extraction a very difficult task"
    res.loc[(res["ReportName"] == "Abbvie_2019_en.pdf"), "ms_comment_man"] = "Challenge: No absolute carbon emissions included, but p. 31 mentions percent change in absolute carbon emissions as compared to 2015"
    res.loc[(res["ReportName"] == "eon_2010_en.pdf"), "ms_comment_man"] = "Challenge: data is on three separate pages"
    res.loc[(res["ReportName"] == "volkswagen_2019_en.pdf"), "ms_comment_man"] = "Challenges: The report has four tables on a single page: Scope 1 emissions (in kg/vehivle); Scope 1 emissions (in million tonnes/year); Scope 1 + 2 emissions (in kg/vehicle); Scope 1 + Scope 2 emissions (in million tonnes/year). It would be possible to CALCULATE Scope 2 emissions from this, but the manual coder didn't do it. The subsequent page contains a detailed calculation of Scope 3 emissions"
    
    if print_to_excel:
        res.to_excel(os.path.join(path_to_data, "processed-data", "data_labeled_anon_as_grid.xlsx"))

    return res
