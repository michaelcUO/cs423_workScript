from __future__ import annotations  #must be first line in your library!
import warnings
import pandas as pd
import numpy as np
import types
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices



class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result






class CustomOHETransformer(BaseEstimator, TransformerMixin):
    """
    Custom One-Hot Encoding Transformer using pandas.get_dummies.

    Encodes specified column(s) using one-hot encoding.
    """

    def __init__(self, target_column: str) -> None:
        """
        Parameters
        ----------
        target_column : str
            The column to one-hot encode.
        """
        assert isinstance(target_column, str), f'{self.__class__.__name__} expected str but got {type(target_column)} instead.'
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - does nothing for this transformer.

        Returns
        ----------
        self : CustomOHETransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode specified columns in the input DataFrame.

        Returns
        ----------
        pd.DataFrame
            Transformed DataFrame with specified columns one-hot encoded.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

        assert self.target_column in X.columns, f'{self.__class__.__name__}.transform unknown column {self.target_column}'

        X_encoded = pd.get_dummies(X,
                                   columns=[self.target_column],
                                   dummy_na=False,
                                   drop_first=False,
                                   dtype=int)
        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Combine fit and transform.

        Returns
        ----------
        pd.DataFrame
            Transformed DataFrame with specified columns one-hot encoded.
        """
        return self.transform(X)
    
    
    
    
    
    
    
    
    
class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    #your code below

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - does nothing for this transformer.

        This method is required by the scikit-learn transformer interface but doesn't
        perform any actual fitting operation for this specific transformer.
        It simply prints a warning message and returns itself.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomDropColumnsTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the column dropping or keeping operation to the input DataFrame.

        This method performs the core functionality of the transformer by either
        dropping or keeping specified columns based on the 'action' parameter
        set during initialization.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the columns to operate on.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with the specified columns either dropped or kept.

        Raises
        ------
        AssertionError
            - If X is not a pandas DataFrame.
            - If action is 'keep' and any columns in column_list are not found in X.
        """
        assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

        missing_cols = set(self.column_list) - set(X.columns)
        if self.action == 'keep':
            assert not missing_cols, f'{self.__class__.__name__}.transform unknown columns to keep: {list(missing_cols)}'
            X_ = X[self.column_list]

        elif self.action == 'drop':
            if missing_cols:
                print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {list(missing_cols)}\n") # Warning for missing columns in 'drop'.
            # assert not missing_cols, f'{self.__class__.__name__}.transform unknown columns to drop: {list(missing_cols)}'
            X_ = X.drop(columns=self.column_list, errors='ignore') # Ignore errors for missing columns.

        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        return self.transform(X)


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """
    def __init__(self, target_column):
        self.target_column = target_column
        self.low_wall = None
        self.high_wall = None

    def fit(self, X: pd.DataFrame, y=None):
      assert isinstance(X, pd.DataFrame), (
          f"{self.__class__.name__}fit expected a DataFrame, got {type(X)}"
      )

      assert self.target_column in X.columns, (
          f"{self.__class__.__name__}.fit unknown column '{self.target_column}'"
      )

      assert pd.api.types.is_numeric_dtype(X[self.target_column]), (
          f"{self.__class__.__name__}.fit expected numeric dtype in '{self.target_column}'"
      )

      # Computer mean and std.
      m = X[self.target_column].mean()
      sigma = X[self.target_column].std()

      # Compute the low and high walls.
      self.low_wall = m - 3 * sigma
      self.high_wall = m + 3 * sigma

      return self

    def transform(self, X: pd.DataFrame):
        assert self.low_wall is not None and self.high_wall is not None, (
            f"{self.__class__.__name__}.transform called before fit. "
        )
        
        assert isinstance(X, pd.DataFrame), (
            f"{self.__class__.name__}transform expected a DataFrame, got {type(X)}"
        )

        # Clip and reset index.

        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
        return X_.reset_index(drop=True)



class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """

    def __init__(self, target_column, fence='outer'):
        assert fence in ['inner', 'outer'], f"fence must be 'inner' or 'outer', got {fence}"
        self.target_column = target_column
        self.fence = fence
        self.inner_low = None
        self.inner_high = None
        self.outer_low = None
        self.outer_high = None

    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame), (
            f"{self.__class__.name__}.fit expected a DataFrame, got {type(X)}"
        )

        assert self.target_column in X.columns, (
            f"{self.__class__.__name__}.fit unknown column '{self.target_column}'"
        )

        assert pd.api.types.is_numeric_dtype(X[self.target_column]), (
            f"{self.__class__.__name__}.fit expected numeric dtype in '{self.target_column}'"
        )

        col = X[self.target_column]
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1

        self.inner_low = q1 - 1.5 * iqr
        self.inner_high = q3 + 1.5 * iqr
        self.outer_low = q1 - 3.0 * iqr
        self.outer_high = q3 + 3.0 * iqr

        return self


    def transform(self, X: pd.DataFrame):
        assert self.inner_low is not None, (
            f"{self.__class__.__name__}.transform called before fit. "
        )
        
        assert isinstance(X, pd.DataFrame), (
            f"{self.__class__.name__}.transform expected a DataFrame, got {type(X)}"
        )

        lower = self.inner_low if self.fence == 'inner' else self.outer_low
        upper = self.inner_high if self.fence == 'inner' else self.outer_high

        Xc = X.copy()
        Xc[self.target_column] = Xc[self.target_column].clip(lower=lower, upper=upper)
        return Xc.reset_index(drop=True)
        



#first define the pipeline (but do not invoke it)
titanic_transformer = Pipeline(steps=[
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    #add your new ohe step below
    ('joined_ohe', CustomOHETransformer(target_column='Joined')),
    ('fare', CustomTukeyTransformer(target_column='Fare', fence='outer'))
], verbose=True)


#Once you finish challenge 3 you will have first step - figure out others - I ended up with 5 total steps
customer_transformer = Pipeline(steps=[
    #fill in the steps on your own
    ('drop_columns', CustomDropColumnsTransformer(['ID'], 'drop')),
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('experience_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high': 2})),
    ('os', CustomOHETransformer(target_column='OS')),
    ('isp', CustomOHETransformer(target_column='ISP')),
    ('time spent', CustomTukeyTransformer('Time Spent', 'inner'))
], verbose=True)