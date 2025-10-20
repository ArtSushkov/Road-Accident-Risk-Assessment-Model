# === Импорт библиотек ===

# Импорты стандартной библиотеки Python
from typing import List, Optional, Union  # Для аннотаций типов

# Импорты сторонних библиотек для работы с данными и вычислений
import numpy as np  # Математические вычисления
import pandas as pd  # Работа с табличными данными

# Импорты для визуализации
import matplotlib.pyplot as plt  # Построение графиков
import seaborn as sns  # Статистическая визуализация
from matplotlib.patches import Patch  # Создание патчей для легенд

# Импорты для Jupyter окружения
from IPython.display import HTML, display  # Отображение HTML в ноутбуках

# Импорты для предобработки данных и ML
from sklearn.base import BaseEstimator, TransformerMixin  # Базовые классы sklearn
from sklearn.compose import ColumnTransformer  # Преобразование столбцов
from sklearn.impute import SimpleImputer  # Заполнение пропусков
from sklearn.pipeline import Pipeline  # Создание пайплайнов
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Кодирование и масштабирование
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold  # CV и поиск параметров
from category_encoders import CatBoostEncoder  # CatBoost кодирование категорий

# Импорты для статистического анализа
import statsmodels.api as sm  # Статистические модели
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF анализ
from phik.report import plot_correlation_matrix  # Матрица корреляции Фи-К


# === Задание констант и списков признаков ===

# константа RANDOM_STATE
RANDOM_STATE = 42

# константа TEST_SIZE
TEST_SIZE = 0.25

# Списки признаков для разных типов обработки
binary_features = ['cellphone_in_use', 'vehicle_transmission']  # Бинарные признаки
ohe_features = ['vehicle_type', 'road_surface']  # Признаки для One-Hot Encoding
catboost_features = ['weather_1', 'lighting', 'direction', 'road_condition_1', 'county_city_location']  # Признаки для CatBoostEncoder
numeric_features = ['distance', 'insurance_premium', 'vehicle_age']  # Числовые признаки


# === Собственные функции и класс ===

def extended_describe(df, name="DataFrame"):
    """
    Возвращает расширенное описание датафрейма в виде HTML-таблицы.

    Добавляет полезные метрики к стандартному описанию:
    - mismatch%: отклонение среднего (mean) от медианы (50%) в процентах
    - rel_std%: относительное стандартное отклонение в процентах (std / mean * 100)
    - cat_top_ratio%: доля самого частого значения в категориальных столбцах (freq / count * 100)

    Параметры:
    ----------
    df : pd.DataFrame
        Входной датафрейм для анализа.
    name : str, optional (default="DataFrame")
        Название датафрейма, отображается в заголовке вывода.

    Возвращает:
    -----------
    None
        Результат выводится в виде HTML-таблицы в интерфейсе (например, Jupyter Notebook).

    Пример использования:
    ---------------------
    >>> extended_describe(messages_df, name="messages_df")

    Числовое описание данных: messages_df
    +--------------+--------+--------+----------+-------------+----------------+----------------+
    |              | count  | unique |   mean   |     std     |   mismatch%    |   rel_std%     |
    +--------------+--------+--------+----------+-------------+----------------+----------------+
    | column_1     | 100.00 | NaN    |  50.23   |   10.45     |     0.46       |     20.81      |
    | category_col | 100.00 | 5.00   |   NaN    |    NaN      |     NaN        |     NaN        |
    +--------------+--------+--------+----------+-------------+----------------+----------------+

    Примечание:
    ------------
    Функция работает как с числовыми, так и с категориальными столбцами.
    Требует библиотеки pandas и IPython для отображения HTML.
    """

    # Получаем стандартное описание
    desc = df.describe(include='all')

    # Создаем копию, чтобы не модифицировать исходный describe
    pivot = desc.copy()

    # Добавляем mismatch% для числовых столбцов
    if 'mean' in pivot.index and '50%' in pivot.index:
        with pd.option_context('mode.use_inf_as_na', True):
            mismatch = ((pivot.loc['mean'] - pivot.loc['50%']) / pivot.loc['50%']) * 100
        mismatch.name = 'mismatch%'
        pivot = pd.concat([pivot, mismatch.to_frame().T])

    # Добавляем rel_std% для числовых столбцов
    if 'std' in pivot.index and 'mean' in pivot.index:
        with pd.option_context('mode.use_inf_as_na', True):
            rel_std = (pivot.loc['std'] / pivot.loc['mean']) * 100
        rel_std.name = 'rel_std%'
        pivot = pd.concat([pivot, rel_std.to_frame().T])

    # Добавляем cat_top_ratio% для категориальных столбцов
    if 'freq' in pivot.index and 'count' in pivot.index:
        with pd.option_context('mode.use_inf_as_na', True):
            cat_ratio = (pivot.loc['freq'] / pivot.loc['count']) * 100
        cat_ratio.name = 'cat_top_ratio%'
        pivot = pd.concat([pivot, cat_ratio.to_frame().T])

    # Округляем и транспонируем
    styled_table = pivot.round(2).T

    # Выводим в HTML
    print(f'\033[1mЧисловое описание данных: {name}\033[0m')
    display(HTML(styled_table.to_html(index=True)))


def plot_distribution_with_boxplot(df, features, target_col, category_order=None, bins=None,
                                 auto_bins=False, log_scale=False, minor_category_threshold=0.05,
                                 show_category_types=True):
    """Визуализирует распределения признаков с разделением по категориям целевой переменной.

    Автоматически преобразует числовые целевые переменные в категориальные,
    если они содержат ≤10 уникальных значений. Это преобразование не влияет на исходный датафрейм.

    Создает сетку графиков:
    - Для каждого признака отображаются 2 строки:
      1) Гистограмма с наложениями категорий (stacked) и KDE-кривой (с возможностью
         использования второй оси Y для редких категорий, если они есть)
      2) Горизонтальный boxplot, где категории размещены по оси y
    - Цвета категорий согласованы между графиками и указаны в легенде
    - Для дисбалансированных данных редкие категории (ниже порога) отображаются на отдельной оси Y
    - Если малых категорий нет, вторая ось Y не создается, а подпись "Основная" в легенде не добавляется
    - Графики автоматически размещаются в сетке (до 4 столбцов)
    - Убираются пустые оси для незаполненных ячеек сетки
    - Настройки: сетка, поворот меток, оптимизация макета

    Args:
        df (pd.DataFrame): DataFrame с данными
        features (List[Tuple[str, str]]): Список кортежей (колонка, человекочитаемая метка)
        target_col (str): Название целевой колонки. Если тип числовой и содержит ≤10 уникальных значений,
                          будет преобразована в категориальную внутри функции
        category_order (Optional[List[str]]): Список категорий в нужном порядке. Если None,
            используется отсортированный порядок (по умолчанию None)
        bins (Optional[int]): Количество корзин для гистограмм. Если None и auto_bins=False,
            используется значение по умолчанию в sns.histplot (по умолчанию None)
        auto_bins (bool): Если True, для каждого признака количество корзин будет определяться
            автоматически с помощью правила Фридмана-Дьякониса (по умолчанию False)
        log_scale (bool or str or list): Если True, логарифмический масштаб применяется ко всем признакам.
            Если 'auto', применяется только к признакам с большим разбросом значений.
            Если список, указывает к каким именно признакам применить (по именам колонок).
            (по умолчанию False)
        minor_category_threshold (float): Порог для определения редких категорий (доля от общего числа записей).
            Категории с долей меньше этого значения будут отображаться на отдельной оси Y.
            (по умолчанию 0.05)
        show_category_types (bool): Если True, в легенде будет указано, является ли категория
            основной или редкой. **Если малых категорий нет, пометка "Основная" не добавляется.**
            (по умолчанию True)

    Notes:
        - features ожидает кортежи (колонка, человекочитаемая метка)
        - Для числовых целевых переменных с ≤10 уникальными значениями автоматически создаётся
          категориальная переменная (без изменения исходного DataFrame)
        - Если нет малых категорий, правая ось Y не создается, а легенда не содержит избыточных меток

    Examples:
        >>> # Пример 1: Базовое использование с автоматическим определением категорий
        >>> features = [('age', 'Возраст'), ('income', 'Доход')]
        >>> plot_distribution_with_boxplot(df, features, 'gender')

        >>> # Пример 2: Логарифмический масштаб для всех признаков
        >>> plot_distribution_with_boxplot(df, features, 'gender', log_scale=True)

        >>> # Пример 3: Логарифмический масштаб только для определенных признаков
        >>> plot_distribution_with_boxplot(df, features, 'gender', log_scale=['income'])

        >>> # Пример 4: Автоматическое определение признаков для логарифмического масштаба
        >>> plot_distribution_with_boxplot(df, features, 'gender', log_scale='auto')

        >>> # Пример 5: Настройка отображения редких категорий (менее 10% записей)
        >>> plot_distribution_with_boxplot(df, features, 'rare_category', minor_category_threshold=0.1)

        >>> # Пример 6: Отключение пометок категорий в легенде
        >>> plot_distribution_with_boxplot(df, features, 'category', show_category_types=False)

        >>> # Пример 7: Комбинированное использование параметров
        >>> plot_distribution_with_boxplot(
        ...     df,
        ...     features=[('age', 'Возраст'), ('income', 'Доход')],
        ...     target_col='membership_type',
        ...     minor_category_threshold=0.1,
        ...     log_scale='auto',
        ...     auto_bins=True
        ... )
    """

    # Проверка и преобразование целевой переменной
    target_series = df[target_col]
    unique_count = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)

    # Создаем временную целевую переменную
    if is_numeric and unique_count <= 10:
        transformed_target = target_series.astype(str)
        print(f"Целевой признак '{target_col}' преобразован в категориальный. "
              f"Уникальные значения: {sorted(transformed_target.unique())}")
    else:
        transformed_target = target_series.copy()

    # Получаем категории в нужном порядке (удалив строки с пропусками, если они имеются)
    if category_order is None:
        if is_numeric and unique_count <= 10:
            numeric_categories = sorted(df[target_col].dropna().unique())
            categories = [str(val) for val in numeric_categories]
        else:
            categories = sorted(transformed_target.dropna().unique())
    else:
        categories = category_order

    # Определяем основные и второстепенные категории
    category_counts = transformed_target.value_counts(normalize=True)
    major_categories = category_counts[category_counts >= minor_category_threshold].index.tolist()
    minor_categories = category_counts[category_counts < minor_category_threshold].index.tolist()

    # Подготовка цветовой палитры
    palette = plt.cm.Paired(np.linspace(0, 1, len(categories)))
    category_to_color = {cat: color for cat, color in zip(categories, palette)}

    # Формируем элементы легенды с указанием типа категории (если нужно и если есть малые категории)
    has_minor_categories = len(minor_categories) > 0
    should_show_types = show_category_types and has_minor_categories

    if should_show_types:
        legend_elements = [
            Patch(facecolor=color,
                 label=f"{cat} (Основная)" if cat in major_categories else f"{cat} (Малая)",
                 alpha=0.6)
            for cat, color in category_to_color.items()
        ]
    else:
        legend_elements = [
            Patch(facecolor=color, label=str(cat), alpha=0.6)
            for cat, color in category_to_color.items()
        ]

    # Настройка сетки графиков
    n_features = len(features)
    ncols = min(4, n_features)
    rows_per_feature = 2
    nrows = (n_features + ncols - 1) // ncols * rows_per_feature

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 3 * nrows),
        squeeze=False
    )

    # Функция для вычисления оптимального числа корзин по правилу Фридмана-Дьякониса
    def calculate_fd_bins(data):
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / (len(data) ** (1/3))
        if bin_width == 0:
            return 30
        return int(np.ceil((data.max() - data.min()) / bin_width))

    # Определяем, к каким признакам применять логарифмический масштаб
    if log_scale == 'auto':
        log_features = []
        for feature_col, _ in features:
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                data = df[feature_col].dropna()
                if len(data) > 0 and data.min() > 0 and (data.max() / data.min() > 100):
                    log_features.append(feature_col)
        if log_features:
            print(f"Автоматически применен логарифмический масштаб к признакам: {log_features}")
    elif log_scale is True:
        log_features = [feature_col for feature_col, _ in features
                       if pd.api.types.is_numeric_dtype(df[feature_col])]
    elif isinstance(log_scale, list):
        log_features = log_scale
    else:
        log_features = []

    # Цикл по признакам для создания графиков
    for i, (feature_col, feature_label) in enumerate(features):
        col = i % ncols
        row_base = (i // ncols) * rows_per_feature

        # Определение количества корзин для текущего признака
        current_bins = bins
        if auto_bins and pd.api.types.is_numeric_dtype(df[feature_col]):
            data_clean = df[feature_col].dropna()
            if len(data_clean) > 0:
                current_bins = calculate_fd_bins(data_clean)
                print(f"Для признака '{feature_col}' автоматически выбрано {current_bins} корзин")

        use_log_scale = feature_col in log_features

        # Гистограмма
        ax_hist = axes[row_base, col]
        ax_hist_right = None  # Инициализируем как None

        # Рисуем основные категории на левой оси
        if major_categories:
            sns.histplot(
                data=df[df[target_col].isin(major_categories)],
                x=feature_col,
                hue=transformed_target,
                kde=True,
                multiple='stack',
                palette=category_to_color,
                ax=ax_hist,
                legend=False,
                bins=current_bins,
                log_scale=use_log_scale
            )

        # Только если есть малые категории — создаём правую ось и рисуем на ней
        if minor_categories:
            ax_hist_right = ax_hist.twinx()

            sns.histplot(
                data=df[df[target_col].isin(minor_categories)],
                x=feature_col,
                hue=transformed_target,
                kde=True,
                multiple='stack',
                palette=category_to_color,
                ax=ax_hist_right,
                legend=False,
                bins=current_bins,
                log_scale=use_log_scale
            )

            # Настраиваем правую ось
            ax_hist_right.spines['right'].set_color('gray')
            ax_hist_right.tick_params(axis='y', colors='gray')
            ax_hist_right.yaxis.label.set_color('gray')
            ax_hist_right.set_ylabel(
                f'Малые категории (<{minor_category_threshold*100:.0f}%)',
                color='gray',
                fontsize=8
            )

        # Настройка левой оси (основной)
        ax_hist.set_title(f'Распределение {feature_label}', fontsize=10)
        ax_hist.set_xlabel('')

        if use_log_scale:
            ax_hist.set_xscale('log')
            ax_hist.set_title(f'Распределение {feature_label} (log scale)', fontsize=10)

        # Подпись оси Y: только если нет малых категорий или если это первый график
        if minor_categories:
            if i == 0:
                ax_hist.set_ylabel(
                    f'Основные категории (≥{minor_category_threshold*100:.0f}%)',
                    fontsize=9
                )
            else:
                ax_hist.set_ylabel('')
        else:
            # Если малых категорий нет, просто подписываем как "Частота"
            if i == 0:
                ax_hist.set_ylabel('Частота', fontsize=9)
            else:
                ax_hist.set_ylabel('')

        # Легенда только на первом графике
        if i == 0:
            ax_hist.legend(
                handles=legend_elements,
                title=target_col,
                fontsize=8,
                title_fontsize=8,
                loc='upper right'
            )

        ax_hist.grid(axis='y', alpha=0.3)
        ax_hist.tick_params(axis='x', labelrotation=0)

        # Boxplot
        ax_box = axes[row_base + 1, col]
        sns.boxplot(
            data=df,
            x=feature_col,
            y=transformed_target,
            order=categories,
            palette=category_to_color,
            orient='h',
            ax=ax_box,
            width=0.6
        )

        if use_log_scale:
            ax_box.set_xscale('log')

        # Настройка boxplot
        if i == 0:
            ax_box.tick_params(axis='y', labelrotation=45, labelsize=8)
        else:
            ax_box.set_yticklabels([])

        ax_box.set_xlabel(feature_label, fontsize=9)
        ax_box.grid(axis='y', alpha=0.3)
        ax_box.tick_params(axis='x', labelrotation=0)
        ax_box.set_ylabel('')

    # Скрытие пустых осей
    for row in range(nrows):
        for col in range(ncols):
            current_idx = (row // rows_per_feature) * ncols + col
            if current_idx >= n_features:
                axes[row, col].set_visible(False)

    plt.tight_layout(h_pad=1.5, w_pad=1.5)
    plt.show()


def plot_violin_combinations(
    df: pd.DataFrame,
    x_column: Union[str, List[str], None] = None,  # Может быть строкой, списком или None
    y_features: list = None,
    hue_features: list = None,
    max_classes: int = 10,
    figsize_per_plot: tuple = (5, 5),
    numeric_to_categorical: bool = True,
    log_scale_x: bool = False
):
    """
    Строит комбинации violinplot для числовых и категориальных признаков.

    Args:
    -----------
        df : pd.DataFrame
            Входной датафрейм с данными.
        x_column : str или list, optional (default=None)
            Имя числового признака или список признаков для оси X.
            Если None, автоматически выбираются все числовые столбцы.
        y_features : list, optional (default=None)
            Список категориальных признаков для оси Y.
            Если None, определяются автоматически:
            - Объектные столбцы с 2-10 уникальными значениями.
            Если numeric_to_categorical=True, числовые столбцы будут преобразованы в категориальные.
        hue_features : list, optional (default=None)
            Список бинарных категориальных признаков для разделения (hue).
            Если None, определяются автоматически:
            - Столбцы с ровно 2 уникальными значениями (числовые или категориальные).
        max_classes : int, optional (default=10)
            Максимальное количество уникальных значений для признаков на оси Y (по умолчанию до 10).
        figsize_per_plot : tuple, optional (default=(5, 5))
            Размер каждого подграфика в дюймах (width, height).
        numeric_to_categorical : bool, optional (default=True)
            Если True, числовые признаки в y_features будут преобразованы в категориальные.
        log_scale_x : bool, optional (default=False)
            Если True, ось X будет в логарифмическом масштабе.

    Returns:
    --------
    None
        Выводит графики и/или сообщения об ошибках.

    Example:
    --------
        >>> import pandas as pd
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Загрузка тестовых данных
        >>> df = sns.load_dataset('titanic')
        >>>
        >>> # Пример 1: Автоматический режим (все числовые на X)
        >>> plot_violin_combinations(df=df)
        >>>
        >>> # Пример 2: Ручное задание признаков
        >>> plot_violin_combinations(
        ...     df=df,
        ...     x_column=['age', 'fare'],
        ...     y_features=['class', 'embarked'],
        ...     hue_features=['sex', 'alive'],
        ...     figsize_per_plot=(6, 4),
        ...     log_scale_x=True
        ... )
        >>>
        >>> # Пример 3: Частично автоматический режим
        >>> plot_violin_combinations(
        ...     df=df,
        ...     x_column='fare',
        ...     log_scale_x=True
        ... )
    """
    # --- 1. Подготовка данных ---
    temp_df = df.copy()

    # --- 2. Определение x_columns (всех числовых признаков, если не задано) ---
    if x_column is None:
        x_columns = temp_df.select_dtypes(include='number').columns.tolist()
        if not x_columns:
            raise ValueError("Нет числовых столбцов для оси X.")
        print(f"Автоматически выбраны столбцы для X: {x_columns}")
    elif isinstance(x_column, str):
        x_columns = [x_column]
    else:
        x_columns = list(x_column)

    # Проверка, что все x_columns существуют и числовые
    for col in x_columns:
        if col not in temp_df.columns:
            raise ValueError(f"Столбец '{col}' не найден.")
        if not pd.api.types.is_numeric_dtype(temp_df[col]):
            raise ValueError(f"Столбец '{col}' не числовой.")

    # --- 3. Автоматическое определение y_features и hue_features ---
    if y_features is None:
        y_features = []
        for col in temp_df.select_dtypes(include=['object', 'category']).columns:
            if col not in x_columns and 2 <= temp_df[col].nunique(dropna=False) <= max_classes:
                y_features.append(col)

        if numeric_to_categorical:
            for col in temp_df.select_dtypes(include='number').columns:
                if col not in x_columns and 2 <= temp_df[col].nunique(dropna=False) <= max_classes:
                    unique_values = sorted(temp_df[col].dropna().unique())
                    temp_df[col] = temp_df[col].astype(str)
                    temp_df[col] = pd.Categorical(temp_df[col], categories=[str(v) for v in unique_values])
                    y_features.append(col)

    if hue_features is None:
        hue_features = []
        for col in temp_df.columns:
            if col not in x_columns and temp_df[col].nunique(dropna=False) == 2:
                hue_features.append(col)

    # --- 4. Проверка признаков ---
    if not y_features:
        print("Нет подходящих признаков для Y.")
        return
    if not hue_features:
        print("Нет бинарных признаков для hue.")
        return

    # --- 5. Генерация всех комбинаций (x_col, y_col, hue_col) ---
    combinations = []
    seen = set()
    for x_col in x_columns:
        for y_col in y_features:
            for hue_col in hue_features:
                if y_col != hue_col and y_col != x_col and hue_col != x_col:
                    key = tuple(sorted([x_col, y_col, hue_col]))
                    if key not in seen:
                        combinations.append((x_col, y_col, hue_col))
                        seen.add(key)

    if not combinations:
        print("Нет допустимых комбинаций признаков.")
        return

    # --- 6. Построение графиков ---
    total_plots = len(combinations)
    ncols = min(3, total_plots)
    nrows = (total_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per_plot[0], nrows * figsize_per_plot[1]))
    axes = axes.flatten() if total_plots > 1 else [axes]

    for i, (x_col, y_col, hue_col) in enumerate(combinations):
        ax = axes[i]
        try:
            sns.violinplot(
                x=x_col,
                y=y_col,
                hue=hue_col,
                data=temp_df,
                split=True,
                palette='Set2',
                inner='quartile',
                ax=ax
            )
            if log_scale_x:
                ax.set_xscale('log')
                ax.set_xlabel(f'{x_col} (log scale)')
            ax.set_title(f"X: {x_col}\nY: {y_col} | Hue: {hue_col}")
            ax.legend(title=hue_col, loc='upper right')
        except Exception as e:
            ax.set_title(f"Ошибка: {x_col}, {y_col}, {hue_col}")
            ax.text(0.5, 0.5, str(e), ha='center', va='center')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame,
    target_col: str,
    categorical_features: Optional[List[str]] = None,
    log_scale: bool = False,
    ncols: int = 5,
    max_categories: int = 10,
    rotate_labels: int = 45,
    label_fontsize: int = 10,
    wrap_labels: bool = True,
    label_max_length: int = 15,
    figsize_width: float = 5.0,
    figsize_height: float = 3.5,
    annotation_pos: str = 'top'  # 'top' или 'above_title'
) -> None:
    """
    Визуализирует распределение категориальных признаков относительно бинарного целевого признака.
    Для каждого категориального признака строит два графика:
    1. Абсолютные значения (countplot)
    2. Относительные доли (stacked barplot)

    Parameters
    ----------
    df : pd.DataFrame
        Входной DataFrame с данными для визуализации.
    target_col : str
        Название целевого бинарного признака (должен содержать ровно 2 уникальных значения).
    categorical_features : List[str], optional
        Список категориальных признаков для анализа. Если None, будут использованы все столбцы
        с типами 'object' и 'category', исключая целевой признак.
    log_scale : bool, default=False
        Если True, ось Y на графиках абсолютных значений будет в логарифмическом масштабе.
    ncols : int, default=5
        Количество столбцов в сетке графиков.
    max_categories : int, default=10
        Максимальное количество категорий для отображения на одном графике.
        Если в признаке больше категорий, отображаются только top-N по частоте.
    rotate_labels : int, default=45
        Угол поворота подписей категорий на оси X (в градусах).
    label_fontsize : int, default=10
        Размер шрифта для подписей категорий на оси X.
    wrap_labels : bool, default=True
        Если True, длинные метки категорий будут переноситься на несколько строк.
    label_max_length : int, default=15
        Максимальная длина подписи категории до переноса на новую строку (если wrap_labels=True).
    figsize_width : float, default=5.0
        Ширина одного подграфика в дюймах.
    figsize_height : float, default=3.5
        Высота одного подграфика в дюймах.
    annotation_pos : str, default='top'
        Положение аннотации о количестве категорий:
        - 'top' - в правом верхнем углу графика
        - 'above_title' - над заголовком графика

    Raises
    ------
    ValueError
        Если целевой столбец отсутствует в DataFrame или не является бинарным.
    KeyError
        Если указанные категориальные признаки отсутствуют в DataFrame.
    ValueError
        Если не найдено категориальных признаков для визуализации.

    Examples
    --------
        >>> # Пример 1: Базовый вызов с автоматическим определением категориальных признаков
        >>> plot_categorical_distributions(
        ...     df=data,
        ...     target_col='target'
        ... )

        >>> # Пример 2: Указание конкретных категориальных признаков
        >>> plot_categorical_distributions(
        ...     df=data,
        ...     target_col='is_default',
        ...     categorical_features=['education', 'family_status', 'income_type', 'age_group'],
        ...     ncols=2
        ... )

        >>> # Пример 3: Использование логарифмической шкалы и настройка внешнего вида
        >>> plot_categorical_distributions(
        ...     df=data,
        ...     target_col='churn',
        ...     log_scale=True,
        ...     rotate_labels=90,
        ...     figsize_width=6,
        ...     figsize_height=4,
        ...     annotation_pos='above_title'
        ... )

        >>> # Пример 4: Ограничение количества категорий и перенос длинных меток
        >>> plot_categorical_distributions(
        ...     df=data,
        ...     target_col='response',
        ...     max_categories=5,
        ...     wrap_labels=True,
        ...     label_max_length=10,
        ...     label_fontsize=8
        ... )

        >>> # Пример 5: Комплексный пример с настройкой всех параметров
        >>> plot_categorical_distributions(
        ...     df=data,
        ...     target_col='loan_status',
        ...     categorical_features=['region', 'product_type', 'client_segment'],
        ...     log_scale=False,
        ...     ncols=3,
        ...     max_categories=8,
        ...     rotate_labels=60,
        ...     label_fontsize=9,
        ...     wrap_labels=True,
        ...     label_max_length=12,
        ...     figsize_width=4.5,
        ...     figsize_height=3.0,
        ...     annotation_pos='top'
        ... )
    """
    # Проверка наличия целевого столбца
    if target_col not in df.columns:
        raise ValueError(f"Целевой столбец '{target_col}' отсутствует в датафрейме.")

    # Проверка, что целевой столбец бинарный
    unique_values = df[target_col].dropna().unique()
    if len(unique_values) != 2:
        raise ValueError(f"Целевой столбец '{target_col}' должен быть бинарным (иметь ровно 2 уникальных значения).")

    # Автоматическое определение категориальных признаков, если не переданы
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_features = [col for col in categorical_features if col != target_col]
    else:
        missing_cols = [col for col in categorical_features if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Следующие категориальные признаки отсутствуют в датафрейме: {missing_cols}")

    if not categorical_features:
        raise ValueError("Не найдено категориальных признаков для отображения.")

    def process_label(label):
        if not isinstance(label, str):
            return str(label)
        if wrap_labels and len(label) > label_max_length:
            return '\n'.join(wrap(label, label_max_length))
        return label

    n_features = len(categorical_features)
    nrows = n_features * 2
    ncols = min(ncols, n_features)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize_width, nrows * figsize_height),
        squeeze=False,
        constrained_layout=True
    )
    fig.suptitle('Распределение категориальных признаков (абсолютные значения и доли)',
                fontsize=16, y=1.02)

    for i, feature in enumerate(categorical_features):
        col = i % ncols
        row_abs = (i // ncols) * 2
        row_rel = row_abs + 1

        value_counts = df[feature].value_counts()
        if len(value_counts) > max_categories:
            top_categories = value_counts.nlargest(max_categories).index
            df_plot = df[df[feature].isin(top_categories)].copy()
            categories_dropped = len(value_counts) - max_categories
        else:
            df_plot = df.copy()
            categories_dropped = 0

        categories = df_plot[feature].value_counts().index
        processed_labels = [process_label(str(cat)) for cat in categories]

        # График абсолютных значений
        sns.countplot(
            data=df_plot,
            x=feature,
            hue=target_col,
            ax=axs[row_abs, col],
            order=categories
        )

        axs[row_abs, col].set_title(f'Абсолютные значения: {feature}', pad=10)
        axs[row_abs, col].set_xticks(range(len(categories)))
        axs[row_abs, col].set_xticklabels(
            processed_labels,
            rotation=rotate_labels,
            ha='right' if rotate_labels > 0 else 'center',
            fontsize=label_fontsize
        )
        axs[row_abs, col].legend(title=target_col, loc='upper right')
        axs[row_abs, col].grid(axis='y', linestyle='--', alpha=0.7)

        if log_scale:
            axs[row_abs, col].set_yscale('log')
            axs[row_abs, col].set_ylabel('Количество (log scale)')
        else:
            axs[row_abs, col].set_ylabel('Количество')

        # Аннотация для абсолютных значений
        if categories_dropped > 0:
            if annotation_pos == 'top':
                axs[row_abs, col].annotate(
                    f'Top {max_categories}/{len(value_counts)}',
                    xy=(0.8, 0.95),
                    xycoords='axes fraction',
                    ha='right',
                    va='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
                )
            else:
                axs[row_abs, col].annotate(
                    f'Top {max_categories} из {len(value_counts)} категорий',
                    xy=(0.5, 1.13),
                    xycoords='axes fraction',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
                )

        # График долей
        grouped = (
            df_plot
            .groupby([feature, target_col])
            .size()
            .unstack(fill_value=0)
            .apply(lambda x: x / x.sum(), axis=1)
            .loc[categories]
        )

        grouped.plot(
            kind='bar',
            stacked=True,
            ax=axs[row_rel, col],
            colormap='viridis'
        )

        axs[row_rel, col].set_title(f'Доли: {feature}', pad=10)
        axs[row_rel, col].set_xticks(range(len(categories)))
        axs[row_rel, col].set_xticklabels(
            processed_labels,
            rotation=rotate_labels,
            ha='right' if rotate_labels > 0 else 'center',
            fontsize=label_fontsize
        )
        axs[row_rel, col].set_ylabel('Доля')
        axs[row_rel, col].legend(title=target_col, loc='upper right')
        axs[row_rel, col].grid(axis='y', linestyle='--', alpha=0.7)

        # Аннотация для долей
        if categories_dropped > 0:
            if annotation_pos == 'top':
                axs[row_rel, col].annotate(
                    f'Top {max_categories}/{len(value_counts)}',
                    xy=(0.8, 0.95),
                    xycoords='axes fraction',
                    ha='right',
                    va='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
                )
            else:
                axs[row_rel, col].annotate(
                    f'Top {max_categories} из {len(value_counts)} категорий',
                    xy=(0.5, 1.13),
                    xycoords='axes fraction',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
                )

    # Удаление пустых подграфиков
    for row in range(nrows):
        for col in range(ncols):
            feature_index = (row // 2) * ncols + col
            if feature_index >= len(categorical_features):
                fig.delaxes(axs[row, col])

    plt.show()


def phik_correlation_matrix(df, target_col=None, threshold=0.9, output_interval_cols=True, interval_cols=None, cell_size=1.1):
    """Строит матрицу корреляции Фи-К (включая целевую переменную) и возвращает корреляции с целевой.

    Args:
        df (pd.DataFrame): DataFrame с данными для анализа
        target_col (str): Название столбца с целевой переменной
        threshold (float): Порог для выделения значимых корреляций (0.9 по умолчанию)
        output_interval_cols (bool): Возвращать ли список числовых непрерывных столбцов
        interval_cols (list): Список числовых непрерывных столбцов (если None, будет определен автоматически)
        cell_size (float): Дюйм на ячейку

    Returns:
        tuple: (correlated_pairs, interval_cols, phi_k_with_target) где:
            - correlated_pairs: DataFrame с парами коррелирующих признаков
            - interval_cols: Список числовых непрерывных столбцов (если output_interval_cols=True)
            - phi_k_with_target: Series с корреляциями признаков с целевой переменной

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from phik import phik_matrix
        >>>
        >>> # Создаем тестовые данные
        >>> data = {
        ...     'price': [100, 200, 150, 300],  # Целевая переменная
        ...     'mileage': [50, 100, 75, 120],
        ...     'brand': ['A', 'B', 'A', 'C'],
        ...     'engine': [1.6, 2.0, 1.8, 2.5]
        ... }
        >>> df = pd.DataFrame(data)
        >>>
        >>> # Анализ корреляций с ручным заданием числовых столбцов
        >>> result = phik_correlation_matrix(df, target_col='price', threshold=0.3, interval_cols=['mileage', 'engine'])
        >>>
        >>> # Получаем результаты:
        >>> correlated_pairs = result[0]  # Пары коррелирующих признаков
        >>> interval_cols = result[1]     # Числовые непрерывные столбцы
        >>> phi_k_with_target = result[2] # Корреляции с ценой
        >>>
        >>> print("Корреляции с ценой:")
        >>> print(phi_k_with_target.sort_values(ascending=False))
    """

    # Определение числовых непрерывных столбцов (если не заданы вручную)
    if interval_cols is None:
        interval_cols = [
            col for col in df.select_dtypes(include=["number"]).columns
            if (df[col].nunique() > 50) or ((df[col] % 1 != 0).any())
        ]

    # Расчет полной матрицы корреляции (включая целевую переменную)
    phik_matrix = df.phik_matrix(interval_cols=interval_cols).round(2)

    # Получение корреляций с целевой переменной
    phi_k_with_target = None
    if target_col is not None and target_col in phik_matrix.columns:
        phi_k_with_target = phik_matrix[target_col].copy()
        # Удаляем корреляцию целевой с собой (всегда 1.0)
        phi_k_with_target.drop(target_col, inplace=True, errors='ignore')

    # Динамическое определение размера фигуры для подстройки размера ячеек
    num_cols = len(phik_matrix.columns)
    num_rows = len(phik_matrix.index)
    cell_size = cell_size  # Дюймов на ячейку
    figsize = (num_cols * cell_size, num_rows * cell_size)

    # Визуализация матрицы
    plot_correlation_matrix(
        phik_matrix.values,
        x_labels=phik_matrix.columns,
        y_labels=phik_matrix.index,
        vmin=0,
        vmax=1,
        color_map="Greens",
        title=r"Матрица корреляции $\phi_K$",
        fontsize_factor=1,
        figsize=figsize
    )
    plt.tight_layout()
    plt.show()

    # Фильтрация значимых корреляций (исключая целевую из пар)
    close_to_one = phik_matrix[phik_matrix >= threshold]
    close_to_one = close_to_one.where(
        np.triu(np.ones(close_to_one.shape), k=1).astype(bool)
    )

    # Удаление строк/столбцов с целевой переменной для анализа пар признаков
    if target_col is not None:
        close_to_one.drop(target_col, axis=0, inplace=True, errors='ignore')
        close_to_one.drop(target_col, axis=1, inplace=True, errors='ignore')

    # Преобразование в длинный формат
    close_to_one_stacked = close_to_one.stack().reset_index()
    close_to_one_stacked.columns = ["признак_1", "признак_2", "корреляция"]
    close_to_one_stacked = close_to_one_stacked.dropna(subset=["корреляция"])

    # Классификация корреляций
    def classify_correlation(corr):
        if corr >= 0.9: return "Очень высокая"
        elif corr >= 0.7: return "Высокая"
        elif corr >= 0.5: return "Заметная"
        elif corr >= 0.3: return "Умеренная"
        elif corr >= 0.1: return "Слабая"
        return "-"

    close_to_one_stacked["класс_корреляции"] = close_to_one_stacked["корреляция"].apply(
        classify_correlation
    )
    close_to_one_sorted = close_to_one_stacked.sort_values(
        by="корреляция", ascending=False
    ).reset_index(drop=True)

    if len(close_to_one_sorted) == 0 and threshold >= 0.9:
        print("\033[1mМультиколлинеарность между парами входных признаков отсутствует\033[0m")

    # Формирование результата
    result = [close_to_one_sorted]
    if output_interval_cols:
        result.append(interval_cols)
    if target_col is not None:
        result.append(phi_k_with_target)
    elif output_interval_cols:
        result.append(None)

    return tuple(result)


def vif(X, font_size=12):
    """Строит столбчатую диаграмму с коэффициентами инфляции дисперсии (VIF) для всех входных признаков.

    Args:
        X (pd.DataFrame): DataFrame с входными признаками для анализа.
        font_size (int): Размер шрифта для текстовых элементов графика (по умолчанию 12).

    Notes:
        - Коэффициент инфляции дисперсии (VIF) показывает степень мультиколлинеарности между признаками.
        - График отображается напрямую через matplotlib.

    Example:
        Пример использования функции:

        >>> import pandas as pd
        >>> from statsmodels.stats.outliers_influence import variance_inflation_factor
        >>> import statsmodels.api as sm
        >>>
        >>> # Создаем тестовый датафрейм
        >>> data = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [2, 4, 6, 8, 10],  # Полностью коррелирует с feature1
        ...     'feature3': [3, 6, 9, 12, 15]   # Частично коррелирует
        ... })
        >>>
        >>> # Вызываем функцию для анализа VIF
        >>> vif(data)
        >>>
        >>> # В результате будет показан график с VIF для каждого признака
        >>> # (feature2 будет иметь очень высокий VIF из-за полной корреляции с feature1)
    """
    # Кодируем категориальные признаки
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

    # Добавляем константу для корректного расчета VIF
    X_with_const = sm.add_constant(X_encoded)

    # Вычисляем VIF для всех признаков, кроме константы (индексы начинаются с 1)
    vif = [variance_inflation_factor(X_with_const.values, i)
           for i in range(1, X_with_const.shape[1])]  # Исключаем константу (0-й столбец)

    # Построение графика с использованием исходных названий признаков (без константы)
    num_features = X_encoded.shape[1]
    fig_width = num_features * 1.2
    fig_height = 12

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(x=X_encoded.columns, y=vif)

    # Настройки графика
    ax.set_ylabel('VIF', fontsize=font_size)
    ax.set_xlabel('Входные признаки', fontsize=font_size)
    plt.title('Коэффициент инфляции дисперсии для входных признаков (VIF)', fontsize=font_size)

    # Метки на осях
    plt.xticks(rotation=90, ha='right', fontsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    # Добавляем значения на столбцы (опционально)
    # ax.bar_label(ax.containers[0], fmt='%.2f', padding=3, fontsize=font_size)

    plt.tight_layout()
    plt.show()


class FeatureUnionWithCatBoost(BaseEstimator, TransformerMixin):
    """Объединяет предобработку основных признаков и CatBoost-кодирование.
    
    Этот класс объединяет несколько этапов предобработки данных:
    - Предобработку числовых, бинарных и категориальных признаков
    - CatBoost-кодирование для указанных категориальных признаков
    
    Attributes:
        preprocessor: Объект предобработки данных (ColumnTransformer).
        catboost_pipe: Pipeline для CatBoost-кодирования.
        catboost_features: Список признаков для CatBoost-кодирования.
        binary_features_count_: Количество бинарных признаков после обработки.
        category_features_count_: Количество категориальных признаков после обработки.
        numeric_features_count_: Количество числовых признаков.
        catboost_features_count_: Количество признаков после CatBoost-кодирования.
    """
    
    def __init__(self, preprocessor=None, catboost_pipe=None, catboost_features=None):
        """Инициализирует объект FeatureUnionWithCatBoost.
        
        Args:
            preprocessor: Объект предобработки данных (ColumnTransformer).
            catboost_pipe: Pipeline для CatBoost-кодирования.
            catboost_features: Список признаков для CatBoost-кодирования.
        """
        self.preprocessor = preprocessor
        self.catboost_pipe = catboost_pipe
        self.catboost_features = catboost_features
        self.binary_features_count_ = 0
        self.category_features_count_ = 0
        self.numeric_features_count_ = 0
        self.catboost_features_count_ = 0

    def fit(self, X, y=None):
        """Обучает преобразователи на данных.
        
        Args:
            X: Входные данные для обучения.
            y: Целевая переменная (опционально).
            
        Returns:
            self: Возвращает экземпляр класса после обучения.
        """
        # Обучаем основной препроцессор
        if self.preprocessor is not None:
            self.preprocessor.fit(X, y)
            
            # Получаем количество фичей после обработки бинарных признаков
            binary_transformer = self.preprocessor.named_transformers_['binary']
            binary_encoder = binary_transformer.named_steps['ohe']
            if hasattr(binary_encoder, 'get_feature_names_out'):
                self.binary_features_count_ = binary_encoder.get_feature_names_out().shape[0]
            elif hasattr(binary_encoder, 'get_feature_names'):
                self.binary_features_count_ = len(binary_encoder.get_feature_names())
            
            # Получаем количество фичей после обработки категориальных признаков
            category_transformer = self.preprocessor.named_transformers_['category']
            category_encoder = category_transformer.named_steps['ohe']
            if hasattr(category_encoder, 'get_feature_names_out'):
                self.category_features_count_ = category_encoder.get_feature_names_out().shape[0]
            elif hasattr(category_encoder, 'get_feature_names'):
                self.category_features_count_ = len(category_encoder.get_feature_names())
            
            # Получаем количество числовых признаков
            for name, transformer, features in self.preprocessor.transformers_:
                if name == 'numeric':
                    self.numeric_features_count_ = len(features)
                    break

        # Обучаем CatBoost-преобразователь
        if self.catboost_pipe is not None and self.catboost_features is not None:
            self.catboost_pipe.fit(X[self.catboost_features], y)
            self.catboost_features_count_ = len(self.catboost_features)
        
        return self

    def transform(self, X):
        """Преобразует входные данные.
        
        Args:
            X: Входные данные для преобразования.
            
        Returns:
            np.ndarray: Преобразованные данные.
        """
        # Применяем основной препроцессор
        X_processed = (
            self.preprocessor.transform(X) 
            if self.preprocessor is not None 
            else np.zeros((X.shape[0], 0))
        )
        
        # Применяем CatBoost-преобразование и объединяем результаты
        if self.catboost_pipe is not None and self.catboost_features is not None:
            X_catboost = self.catboost_pipe.transform(X[self.catboost_features])
            return np.hstack([X_processed, X_catboost])
        
        return X_processed

    def get_feature_counts(self):
        """Возвращает количество признаков каждого типа.
        
        Returns:
            dict: Словарь с количеством признаков каждого типа.
        """
        return {
            'Binary features': self.binary_features_count_,
            'Category features': self.category_features_count_,
            'Numeric features': self.numeric_features_count_,
            'CatBoost features': self.catboost_features_count_,
            'Total features': (
                self.binary_features_count_ + 
                self.category_features_count_ + 
                self.numeric_features_count_ + 
                self.catboost_features_count_
            )
        }

    def get_feature_names_out(self):
        """Возвращает имена выходных признаков.
        
        Returns:
            list: Список имен выходных признаков.
        """
        feature_names = []
        
        # Имена признаков из preprocessor
        if self.preprocessor is not None:
            try:
                # Пытаемся использовать встроенный метод ColumnTransformer
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    feature_names.extend(list(self.preprocessor.get_feature_names_out()))
                else:
                    # Ручная обработка каждого трансформера
                    for name, transformer, features in self.preprocessor.transformers_:
                        if name == 'binary':
                            encoder = transformer.named_steps.get('ohe')
                            if encoder is not None:
                                if hasattr(encoder, 'get_feature_names_out'):
                                    try:
                                        feature_names.extend(
                                            list(encoder.get_feature_names_out(features))
                                        )
                                    except:
                                        feature_names.extend(
                                            list(encoder.get_feature_names(features))
                                        )
                                elif hasattr(encoder, 'get_feature_names'):
                                    feature_names.extend(
                                        list(encoder.get_feature_names(features))
                                    )
                                else:
                                    feature_names.extend(list(features))
                        elif name == 'category':
                            encoder = transformer.named_steps.get('ohe')
                            if encoder is not None:
                                if hasattr(encoder, 'get_feature_names_out'):
                                    try:
                                        feature_names.extend(
                                            list(encoder.get_feature_names_out(features))
                                        )
                                    except:
                                        feature_names.extend(
                                            list(encoder.get_feature_names(features))
                                        )
                                elif hasattr(encoder, 'get_feature_names'):
                                    feature_names.extend(
                                        list(encoder.get_feature_names(features))
                                    )
                                else:
                                    feature_names.extend(list(features))
                        elif name == 'numeric':
                            feature_names.extend(list(features))
            except Exception as e:
                # Fallback если что-то пошло не так
                total_features = self.get_feature_counts()['Total features']
                feature_names = [f'feature_{i}' for i in range(total_features)]
        
        # Имена признаков из catboost_pipe
        if self.catboost_pipe is not None and self.catboost_features is not None:
            # Для CatBoostEncoder пытаемся получить имена через encoder
            catboost_encoder = None
            if hasattr(self.catboost_pipe, 'named_steps') and 'catboost' in self.catboost_pipe.named_steps:
                catboost_encoder = self.catboost_pipe.named_steps['catboost']
            elif hasattr(self.catboost_pipe, 'steps') and len(self.catboost_pipe.steps) > 0:
                # Пытаемся найти encoder в шагах
                for step_name, step_transformer in self.catboost_pipe.steps:
                    if (
                        hasattr(step_transformer, '__class__') and 
                        'catboost' in str(step_transformer.__class__).lower()
                    ):
                        catboost_encoder = step_transformer
                        break
            
            if catboost_encoder is not None and hasattr(catboost_encoder, 'get_feature_names_out'):
                try:
                    feature_names.extend(
                        list(catboost_encoder.get_feature_names_out(self.catboost_features))
                    )
                except:
                    feature_names.extend(list(self.catboost_features))
            elif catboost_encoder is not None and hasattr(catboost_encoder, 'get_feature_names'):
                try:
                    feature_names.extend(
                        list(catboost_encoder.get_feature_names(self.catboost_features))
                    )
                except:
                    feature_names.extend(list(self.catboost_features))
            else:
                feature_names.extend(list(self.catboost_features))
        
        return feature_names

    def set_params(self, **params):
        """Устанавливает параметры для преобразователей.
        
        Args:
            **params: Параметры для установки.
            
        Returns:
            self: Возвращает экземпляр класса.
        """
        # Устанавливаем параметры для preprocessor
        if self.preprocessor is not None:
            preprocessor_params = {
                k[14:]: v for k, v in params.items() 
                if k.startswith('preprocessor__')
            }
            self.preprocessor.set_params(**preprocessor_params)

        # Устанавливаем параметры для catboost_pipe
        if self.catboost_pipe is not None:
            catboost_params = {
                k[15:]: v for k, v in params.items() 
                if k.startswith('catboost_pipe__')
            }
            self.catboost_pipe.set_params(**catboost_params)

        return self


def main(X_train, y_train, model, param_grid, n_splits=3, n_iter=10, weights_train=None):
    """Основная функция для обучения модели с предобработкой данных и гиперпараметрической оптимизацией.
    
    Args:
        X_train: Обучающая выборка признаков.
        y_train: Обучающая выборка целевой переменной.
        model: Модель машинного обучения для обучения.
        param_grid: Словарь с параметрами для поиска.
        n_splits: Количество разбиений для кросс-валидации (по умолчанию 3).
        n_iter: Количество итераций для RandomizedSearchCV (по умолчанию 10).
        weights_train: Веса образцов для обучения (опционально).
        
    Returns:
        tuple: Кортеж из двух элементов:
            - randomized_search: Обученный объект RandomizedSearchCV.
            - mean_fit_time: Среднее время обучения модели.
    """
    # Создание пайплайна для обработки бинарных признаков
    # Используется заполнение пропусков наиболее частыми значениями,
    # OneHotEncoding с удалением бинарных признаков и последующее заполнение константой
    binary_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='if_binary', sparse=False, handle_unknown='error')),
        ('imputer_after', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # Создание пайплайна для обработки категориальных признаков
    # Используется заполнение пропусков наиболее частыми значениями,
    # OneHotEncoding с игнорированием неизвестных значений и последующее заполнение константой
    category_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore')),  
        ('imputer_after', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # Создание пайплайна для обработки числовых признаков
    # Используется заполнение медианой и стандартизация
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Создание главного предобработчика данных
    # Объединяет все три пайплайна для разных типов признаков
    preprocessor = ColumnTransformer([
        ('binary', binary_pipe, binary_features),
        ('category', category_pipe, ohe_features),
        ('numeric', numeric_pipe, numeric_features)
    ], remainder='drop')

    # Создание пайплайна для CatBoostEncoder
    # Используется для кодирования категориальных признаков с учетом целевой переменной
    catboost_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('catboost', CatBoostEncoder())
    ])

    # Создание объединенного обработчика признаков
    # Комбинирует стандартную предобработку и CatBoost-кодирование
    feature_union = FeatureUnionWithCatBoost(
        preprocessor=preprocessor,
        catboost_pipe=catboost_pipe,
        catboost_features=catboost_features
    )

    # Создание финального пайплайна
    # Объединяет предобработку признаков и классификатор
    pipe_final = Pipeline([
        ('feature_union', feature_union),
        ('classifier', model)
    ])

    # Создание внутренней кросс-валидации с использованием стратификации
    # Обеспечивает равномерное распределение классов в каждом фолде
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Определение метрик для оценки качества модели
    # Используются F1-score и accuracy для комплексной оценки
    scoring = {
        'f1': 'f1', 
        'accuracy': 'accuracy'
    }

    # Настройка RandomizedSearchCV для поиска оптимальных гиперпараметров
    # Используется случайный поиск по сетке параметров с кросс-валидацией
    randomized_search = RandomizedSearchCV(
        pipe_final,
        param_grid,
        cv=inner_cv.split(X_train, y_train),
        scoring=scoring,
        refit='f1',  
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_iter=n_iter,
        error_score='raise',
        return_train_score=True
    )

    # Обучение модели с оптимизацией гиперпараметров
    # Если предоставлены веса, они передаются в классификатор
    randomized_search.fit(X_train, y_train, classifier__sample_weight=weights_train)  

    # Вычисление среднего времени обучения модели
    mean_fit_time = np.mean(randomized_search.cv_results_['mean_fit_time'])
    
    # Вывод результатов обучения
    print(f"Mean training time per model: {mean_fit_time:.2f} seconds")
    print("Best parameters:", randomized_search.best_params_)
    print("Best F1-score (CV):", randomized_search.best_score_)
    
    return randomized_search, mean_fit_time