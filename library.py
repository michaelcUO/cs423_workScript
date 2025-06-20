from __future__ import annotations  #must be first line in your library!
import warnings
import pandas as pd
import numpy as np
import types
import datetime
import sklearn
import joblib
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier # From midterm.
from sklearn.model_selection import train_test_split # From midterm.
from sklearn.metrics import f1_score # From midterm.
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV # From Chapter 10.
from sklearn.model_selection import ParameterGrid # From Chapter 11.
from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import HalvingGridSearchCV
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices

# Global constants from chapter 7.
titanic_variance_based_split = 107   #add to your library
customer_variance_based_split = 113  #add to your library


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
                print(f"\nWarning: {self.__class__.__name__} cannot drop these columns as they do not exist in the DataFrame: {list(missing_cols)}\n") # Clearer warning message
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
    def __init__(self, target_column: Hashable) -> None:
        """
        Initialize the CustomSigma3Transformer.

        Parameters
        ----------
        target_column : Hashable
            The name of the column to apply 3-sigma clipping on.
        """
        self.target_column = target_column
        self.low_wall = None
        self.high_wall = None

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit the transformer by computing the 3-sigma bounds.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the target column.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomSigma3Transformer
            Returns self to allow method chaining.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame, if target_column is not in X,
            or if target_column is not numeric.
        """
        assert isinstance(X, pd.DataFrame), (
            f"{self.__class__.__name__}.fit expected a DataFrame, got {type(X)}"
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply 3-sigma clipping to the target column.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the target column.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with 3-sigma clipping applied to the target column.

        Raises
        ------
        AssertionError
            If fit() was not called before transform(), or if X is not a pandas DataFrame.
        """
        assert self.low_wall is not None and self.high_wall is not None, (
            f"{self.__class__.__name__}.transform called before fit. "
        )
        
        assert isinstance(X, pd.DataFrame), (
            f"{self.__class__.__name__}.transform expected a DataFrame, got {type(X)}"
        )

        # Clip and reset index.
        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
        return X_.reset_index(drop=True)

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the target column.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with 3-sigma clipping applied to the target column.
        """
        return self.fit(X, y).transform(X)


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
    >>> transformed_df  # Values clipped according to inner fence
    """

    def __init__(self, target_column: Hashable, fence: Literal['inner', 'outer'] = 'outer') -> None:
        """
        Initialize the CustomTukeyTransformer.

        Parameters
        ----------
        target_column : Hashable
            The name of the column to apply Tukey's fences on.
        fence : Literal['inner', 'outer'], default='outer'
            Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

        Raises
        ------
        AssertionError
            If fence is not 'inner' or 'outer'.
        """
        assert fence in ['inner', 'outer'], f"fence must be 'inner' or 'outer', got {fence}"
        self.target_column = target_column
        self.fence = fence
        self.inner_low = None
        self.inner_high = None
        self.outer_low = None
        self.outer_high = None

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit the transformer by computing Tukey's fences.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the target column.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomTukeyTransformer
            Returns self to allow method chaining.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame, if target_column is not in X,
            or if target_column is not numeric.
        """
        assert isinstance(X, pd.DataFrame), (
            f"{self.__class__.__name__}.fit expected a DataFrame, got {type(X)}"
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
    
    

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Tukey's fences clipping to the target column.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the target column.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with Tukey's fences clipping applied to the target column.

        Raises
        ------
        AssertionError
            If fit() was not called before transform(), or if X is not a pandas DataFrame.
        """
        assert self.inner_low is not None, (
            f"{self.__class__.__name__}.transform called before fit. "
        )
        
        assert isinstance(X, pd.DataFrame), (
            f"{self.__class__.__name__}.transform expected a DataFrame, got {type(X)}"
        )

        lower = self.inner_low if self.fence == 'inner' else self.outer_low
        upper = self.inner_high if self.fence == 'inner' else self.outer_high

        Xc = X.copy()
        Xc[self.target_column] = Xc[self.target_column].clip(lower=lower, upper=upper)
        return Xc.reset_index(drop=True)
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the target column.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with Tukey's fences clipping applied to the target column.
        """
        return self.fit(X, y).transform(X)
    
    
    
#from scratch option

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  """Applies robust scaling to a specified column in a pandas DataFrame.
    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float
        The interquartile range of the target column.
    med : float
        The median of the target column.
  """
  def __init__(self, target_column: str) -> None:
        assert isinstance(target_column, str), \
            f"CustomRobustTransformer expected column name as str, got {type(target_column)}"
        self.target_column = target_column
        self.iqr_: float | None = None
        self.median_: float | None = None

  def fit(self, X: pd.DataFrame, y=None) -> "CustomRobustTransformer":
      assert isinstance(X, pd.DataFrame), \
          f"CustomRobustTransformer.fit expected DataFrame, got {type(X)}"
      assert self.target_column in X.columns, \
          f"CustomRobustTransformer.fit unknown column '{self.target_column}'"

      series = X[self.target_column]
      # Compute Q1, Q3, IQR and median.
      q1 = series.quantile(0.25)
      q3 = series.quantile(0.75)
      self.iqr_ = q3 - q1
      self.median_ = series.median()
      return self

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      assert self.iqr_ is not None and self.median_ is not None, \
          "CustomRobustTransformer.transform called before fit"
      assert isinstance(X, pd.DataFrame), \
          f"CustomRobustTransformer.transform expected DataFrame, got {type(X)}"

      X_scaled = X.copy()
      # If IQR is zero or binary column, skip scaling.
      if self.iqr_ == 0:
          return X_scaled

      X_scaled[self.target_column] = (
          X_scaled[self.target_column] - self.median_
      ) / self.iqr_
      return X_scaled

  def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
      return self.fit(X, y).transform(X)
  
  
  
  
  
class CustomKNNTransformer(BaseEstimator, TransformerMixin):
    """Imputes missing values using KNN.

    This transformer wraps the KNNImputer from scikit-learn and hard-codes
    add_indicator to be False. It also ensures that the input and output
    are pandas DataFrames.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction. Possible values:
        "uniform" : uniform weights. All points in each neighborhood
        are weighted equally.
        "distance" : weight points by the inverse of their distance.
        In this case, closer neighbors of a query point will have a
        greater influence than neighbors which are further away.
    """
    #your code below
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform') -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights, add_indicator=False)

    def fit(self, X: pd.DataFrame, y=None) -> "CustomKNNTransformer":
        """
        Fit the KNN imputer on X.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with missing values.
        y : ignored
            Not used, exists for compatibility.

        Returns
        -------
        CustomKNNTransformer
            The fitted transformer.
        """
        if self.n_neighbors > len(X): # Added to include warning.
          warnings.warn(
              f"Warning: n_neighbors ({self.n_neighbors}) is greater than number of samples ({len(X)}).",
              UserWarning
          )
        self.knn_imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in X.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with missing values.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values imputed.
        """
        return pd.DataFrame(self.knn_imputer.transform(X), columns=X.columns, index=X.index)


############## UPDATED FOR CHAPTER 8. ################
class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    """
    A target encoder that applies smoothing and returns np.nan for unseen categories.

    Parameters:
    -----------
    col: name of column to encode.
    smoothing : float, default=10.0
        Smoothing factor. Higher values give more weight to the global mean.
    """

    def __init__(self, col: str, smoothing: float =10.0):
        self.col = col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_dict_ = None

    # def fit(self, X, y): # BEFORE CHAP8.
    def fit(self, X, y=None):
        """
        Fit the target encoder using training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        if y is None:
            raise ValueError(f"{self.__class__.__name__}.fit requires a target (y), but got None. "
                f"This transformer must be used with fit(X, y), not fit(X) alone.") # NEW: FOR CHAPT8.

        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
        assert self.col in X, f'{self.__class__.__name__}.fit column not in X: {self.col}. Actual columns: {X.columns}'
        assert isinstance(y, Iterable), f'{self.__class__.__name__}.fit expected Iterable but got {type(y)} instead.'
        assert len(X) == len(y), f'{self.__class__.__name__}.fit X and y must be same length but got {len(X)} and {len(y)} instead.'

        #Create new df with just col and target - enables use of pandas methods below
        X_ = X[[self.col]]
        target = self.col+'_target_'
        X_[target] = y

        # Calculate global mean
        self.global_mean_ = X_[target].mean()

        # Get counts and means
        counts = X_[self.col].value_counts().to_dict()    #dictionary of unique values in the column col and their counts
        means = X_[target].groupby(X_[self.col]).mean().to_dict() #dictionary of unique values in the column col and their means

        # Calculate smoothed means
        smoothed_means = {}
        for category in counts.keys():
            n = counts[category]
            category_mean = means[category]
            # Apply smoothing formula: (n * cat_mean + m * global_mean) / (n + m)
            smoothed_mean = (n * category_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            smoothed_means[category] = smoothed_mean

        self.encoding_dict_ = smoothed_means

        return self

    def transform(self, X):
        """
        Transform the data using the fitted target encoder.
        Unseen categories will be encoded as np.nan.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        """

        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.encoding_dict_, f'{self.__class__.__name__}.transform not fitted'

        X_ = X.copy()

        # Map categories to smoothed means, naturally producing np.nan for unseen categories, i.e.,
        # when map tries to look up a value in the dictionary and doesn't find the key, it automatically returns np.nan. That is what we want.
        X_[self.col] = X_[self.col].map(self.encoding_dict_)

        return X_

    def fit_transform(self, X, y):
        """
        Fit the target encoder and transform the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        return self.fit(X, y).transform(X)


def find_random_state(
    features_df: pd.DataFrame,
    labels: Iterable,
    transformer: TransformerMixin,
    n: int = 200
                  ) -> Tuple[int, List[float]]:
    """
    Finds an optimal random state for train-test splitting based on F1-score stability.

    This function iterates through `n` different random states when splitting the data,
    applies a transformation pipeline, and trains a K-Nearest Neighbors classifier.
    It calculates the ratio of test F1-score to train F1-score and selects the random
    state where this ratio is closest to the mean.

    Parameters
    ----------
    features_df : pd.DataFrame
        The feature dataset.
    labels : Union[pd.Series, List]
        The corresponding labels for classification (can be a pandas Series or a Python list).
    transformer : TransformerMixin
        A scikit-learn compatible transformer for preprocessing.
    n : int, default=200
        The number of random states to evaluate.

    Returns
    -------
    rs_value : int
        The optimal random state where the F1-score ratio is closest to the mean.
    Var : List[float]
        A list containing the F1-score ratios for each evaluated random state.

    Notes
    -----
    - If the train F1-score is below 0.1, that iteration is skipped.
    - A higher F1-score ratio (closer to 1) indicates better train-test consistency.
    """

    model = KNeighborsClassifier(n_neighbors=5)
    Var: List[float] = []  # Collect test_f1/train_f1 ratios

    for i in range(n):
        train_X, test_X, train_y, test_y = train_test_split(
            features_df, labels, test_size=0.2, shuffle=True,
            random_state=i, stratify=labels  # Works with both lists and pd.Series
        )

        # Apply transformation pipeline
        transform_train_X = transformer.fit_transform(train_X, train_y)
        transform_test_X = transformer.transform(test_X)

        # Train model and make predictions
        model.fit(transform_train_X, train_y)
        train_pred = model.predict(transform_train_X)
        test_pred = model.predict(transform_test_X)

        train_f1 = f1_score(train_y, train_pred)

        if train_f1 < 0.1:
            continue  # Skip if train_f1 is too low

        test_f1 = f1_score(test_y, test_pred)
        f1_ratio = test_f1 / train_f1  # Ratio of test to train F1-score

        Var.append(f1_ratio)

    mean_f1_ratio: float = np.mean(Var)
    rs_value: int = np.abs(np.array(Var) - mean_f1_ratio).argmin()  # Index of value closest to mean

    return rs_value, Var



############ From Chapter 11. ###########

def halving_search(model, grid, x_train, y_train, factor=3, min_resources="exhaust", scoring='roc_auc'): # Factor changed from 2 to 3 per chapter 11.
  #your code below

      """
      Runs HalvingGridSearchCV on the given model and parameter grid.

      Parameters
      ----------
      model : estimator object
          The machine learning model (e.g., SVC, KNeighborsClassifier).
      grid : dict
          Dictionary of hyperparameters to try.
      x_train : array-like
          Training features.
      y_train : array-like
          Training labels.
      factor : int, default=2
          The ‘halving’ factor.
      min_resources : int or "exhaust", default="exhaust"
          Minimum number of resources to start with.
      scoring : str, default='roc_auc'
          Scoring metric to evaluate model performance.

      Returns
      -------
      grid_result : fitted HalvingGridSearchCV object
          Contains all cross-validation results and best estimator.
      """
      halving_cv = HalvingGridSearchCV(
          model, grid,
          scoring=scoring,
          n_jobs=-1,
          min_resources=min_resources,
          factor=factor,
          cv=5,
          random_state=1234,
          refit=True
      )

      grid_result = halving_cv.fit(x_train, y_train)
      return grid_result




######### From Chapter 11. ###########
#sorts both keys and values
def sort_grid(grid):
  sorted_grid = grid.copy()

  #sort values - note that this will expand range for you
  for k,v in sorted_grid.items():
    sorted_grid[k] = sorted(sorted_grid[k], key=lambda x: (x is None, x))  #handles cases where None is an alternative value

  #sort keys
  sorted_grid = dict(sorted(sorted_grid.items()))

  return sorted_grid




####### From Chapter 10. ###########
def threshold_results(thresh_list, actuals, predicted):
  result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy', 'auc'])

  for t in thresh_list:
      yhat = [1 if v >= t else 0 for v in predicted]

      precision = precision_score(actuals, yhat, zero_division=0)
      recall = recall_score(actuals, yhat, zero_division=0)
      f1 = f1_score(actuals, yhat)
      accuracy = accuracy_score(actuals, yhat)
      auc = roc_auc_score(actuals, predicted)

      result_df.loc[len(result_df)] = {
          'threshold': t,
          'precision': precision,
          'recall': recall,
          'f1': f1,
          'accuracy': accuracy,
          'auc': auc
      }
  result_df = result_df.round(2)

  headers = {
    "selector": "th:not(.index_name)",
    "props": "background-color: #800000; color: white; text-align: center"
  }
  properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

  fancy_df = result_df.style.highlight_max(color = 'pink', axis = 0).format(precision=2).set_properties(**properties).set_table_styles([headers])

  return result_df, fancy_df




########## From Chapter 9. ###########
def dataset_setup(original_table, label_column_name:str, the_transformer, rs, ts=.2):
    #your code below
    labels = original_table[label_column_name].to_list()
    features = original_table.drop(columns=label_column_name)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=ts, shuffle=True, random_state=rs, stratify=labels)

    x_train_transformed = the_transformer.fit_transform(x_train, y_train)
    x_test_transformed = the_transformer.transform(x_test)

    x_train_numpy = x_train_transformed.to_numpy()
    x_test_numpy = x_test_transformed.to_numpy()
    y_train_numpy = np.array(y_train)
    y_test_numpy = np.array(y_test)

    return x_train_numpy, x_test_numpy, y_train_numpy, y_test_numpy






######################### PIPELINES ########################################################

# Actual.
titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', CustomTargetTransformer(col='Joined', smoothing=10)),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer(target_column='Age')),
    ('scale_fare', CustomRobustTransformer(target_column='Fare')),
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)

# Actual.
customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', CustomTargetTransformer(col='ISP')),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer(target_column='Age')), #from 5
    ('scale_time spent', CustomRobustTransformer(target_column='Time Spent')), #from 5
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)


##############################################################################################



########## From Chapter 9. ###########
def titanic_setup(titanic_table, transformer=titanic_transformer, rs=titanic_variance_based_split, ts=.2):
  return dataset_setup(titanic_table, 'Survived', transformer, rs, ts)


########## From Chapter 9. ###########
def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
  return dataset_setup(customer_table, 'Rating', transformer, rs, ts)





