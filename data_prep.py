import pandas as pd

from scipy import stats
from sklearn.utils import resample


def add_dog_info(row, dic):       
    '''
    Extracts informations about the dogs based on the DogID.

    Parameters:
        row (Series): DF-Row from Dog Move Data.
        dic (dict): Contains exisiting information about each individual dog. 
    
    Returns:
        dog (dict): Information about specific dog.
    '''
    
    id = row["DogID"]
    dog = dic[id]
    return dog["Weight"], dog["Age months"], dog["Gender"]


def apply_windowing(df, window_size, step_size):       
    '''
    Combines time series data in definied windows.

    Parameters:
        df (DataFrame): DF with ongoing classification data.
        window_size (int): How many Datapoints should form a window.
        step_size (int): At what intervals the windows should be attached.
    
    Returns:
        df_windows (DataFrame): DF with applied windowing and upsampling.
    '''

    df_windows = pd.DataFrame()

    for column in df.drop(columns=["DogID", "t_sec", "Behavior"]):
        values = []
        for i in range(0, df.shape[0] - window_size, step_size):
            window = df[column].values[i: i + window_size]
            values.append(window)
        df_windows[column] = pd.Series(values).apply(lambda x: x.mean())

    dog_ids = []
    train_labels = []

    for i in range(0, df.shape[0] - window_size, step_size):
        dogid = stats.mode(df['DogID'][i: i + window_size])[0][0]
        label = stats.mode(df['Behavior'][i: i + window_size])[0][0]

        dog_ids.append(dogid)
        train_labels.append(label)

    df_windows["DogID"] = pd.Series(dog_ids)
    df_windows["Behavior"] = pd.Series(train_labels)

    df_windows = sample_up(df_windows)

    return df_windows


def overview_df(df):       
    '''
    Provides an overview of DF.

    Parameters:
        df (DataFrame): Any Data Frame.       
    '''

    print(df.info())
    print(df.head)
    print(df.Behavior.value_counts())

    df.groupby("Behavior").size().plot(kind="pie",
                                       y = "Behavior",
                                       label = "",
                                       autopct="%1.1f%%")


def get_splited_df_dict(df, split_column):       
    '''
    Splits a DF based on values from one column in multiple separated DFs.

    Parameters:
        df (DataFrame): Any Data Frame.
        split_column (str): Name of column by which to separate DF.

    Returns:
        df_dict (dict): Dictionary which contains one separted DF per column value.     
    '''

    df_dict = {value: df[df[split_column] == value] for value in df[split_column].unique()}
    return df_dict


def sample_down(df):       
    '''
    Reduces DataFrame per column value to count of smallest group.

    Parameters:
        df (DataFrame): Any Data Frame.

    Returns:
        df (DataFrame): Downsampled DF.     
    '''

    split_dfs = get_splited_df_dict(df, "Behavior")
    len_jumping = len(split_dfs["Jumping"])
    downsamples = []

    for key in split_dfs.keys():
        downsamples.append(resample(split_dfs[key], replace = True, n_samples = len_jumping))
    
    df = pd.concat(downsamples)
    return df


def sample_up(df):       
    '''
    Expands DataFrame per column value to count of biggest group.

    Parameters:
        df (DataFrame): Any Data Frame.

    Returns:
        df (DataFrame): Upsampled DF.     
    '''

    split_dfs = get_splited_df_dict(df, "Behavior")
    len_lying = len(split_dfs["Lying chest"])
    upsamples = []

    for key in split_dfs.keys():
        upsamples.append(resample(split_dfs[key], replace = True, n_samples = len_lying))
    
    df = pd.concat(upsamples)
    return df    


def unify_behaviors(row):       
    '''
    Unifies three different Behavior columns into one.

    Parameters:
        row (Series): DF-Row from Dog Move Data.

    Returns:
        row[] (str): Behavior from first definied column.     
    '''
    
    if row["Behavior_1"] != "<undefined>":
        return row["Behavior_1"]
    elif row["Behavior_2"] != "<undefined>":
        return row["Behavior_2"]
    else:
        return row["Behavior_3"]


if __name__ == "__main__":
    df_raw = pd.read_csv("DogMoveData.csv")
    #dic_doginfo = pd.read_excel("DogInfo.xlsx", index_col=0).to_dict("index")

    df_raw["Behavior"] = df_raw.apply(lambda row: unify_behaviors(row), axis=1)
    #df_raw["Weight"], df_raw["Age months"], df_raw["Gender"] = df_raw.apply(lambda row: add_dog_info(row, dic_doginfo), axis=1)

    df_clean = df_raw.drop(columns=["TestNum", "Task", "Behavior_1", "Behavior_2", "Behavior_3", "PointEvent"])
    df_clean = df_clean.drop(df_clean[(df_clean.Behavior=="<undefined>") |
                            (df_clean.Behavior=="Synchronization") |
                            (df_clean.Behavior=="Extra_Synchronization") |
                            (df_clean.Behavior=="Bowing")].index)
    df_clean = df_clean.drop_duplicates()
    
    df_windows = apply_windowing(df_clean, 100, 50)

    # df_windows.to_csv("MoveDataWindows.csv", index=False)