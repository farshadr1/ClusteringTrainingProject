import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preProcessing(df: pd.DataFrame, uselessCols: list, dataset_name: str, outlier_threshold:float, scaler='zscore', main_form=None):

    def scalingData(df: pd.DataFrame, scalerMethod: str):
        if scalerMethod == 'None':
            numeric_cols = df.select_dtypes(include='number').columns
            main_form.add_status("-No scaling applied")
            return df, numeric_cols
        elif scalerMethod == 'zscore':
            scalerModel = StandardScaler()
        elif scalerMethod == 'minmax':
            scalerModel = MinMaxScaler()

        numeric_cols = df.select_dtypes(include='number').columns
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scalerModel.fit_transform(df[numeric_cols])
        main_form.add_status(f"-Applied {scalerMethod} scaling to numeric columns")
        return df_scaled , numeric_cols

    def remove_outliers(df: pd.DataFrame, numeric_cols:list):
        df_clean = df.copy()
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)

        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Get records that are outliers in any column
        records_to_remove = outlier_mask.any(axis=1)
        outlier_count = records_to_remove.sum()
        main_form.add_status(f"-Removed {outlier_count} outliers")
        df_clean = df_clean[~records_to_remove]

        return df_clean

    if uselessCols and uselessCols != ['']:
        df.drop(columns=uselessCols, inplace=True)
        main_form.add_status(f"-Dropped columns: {uselessCols}")

    if df.isna().any().any():
        df.dropna(inplace=True)
        main_form.add_status(f"-Dropped NaN values")

    # specific dataset processing:
    if dataset_name == 'User_Knowledge':
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Liver_Disorders':
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Dow_Jones':
        for col in ['close', 'open', 'high', 'low']:
            if col in df.columns:
                df[col] = df[col].replace('[\$,]', '', regex=True).str.strip().astype(float)
            else:
                print(f"Column {col} is empty or not found in the DataFrame.")

        stock_dummies = pd.get_dummies(df['stock'], prefix='stock')
        main_form.add_status('-get_dummies for stock column')
        df = pd.concat([df, stock_dummies], axis=1)
        df.drop(columns=['stock', 'date'], inplace=True)
        main_form.add_status(f"-Dropped columns: {['stock', 'date']}")
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Wholesale_Customers':
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Travel_Reviews':
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Electric_Consumption':
        scaled, numeric_cols = scalingData(df, scaler)

    clean_df = remove_outliers(scaled, numeric_cols=numeric_cols)

    return clean_df, numeric_cols