# 基于 tidymodels 的时间序列预测示例 -------------------------------------------------
# 本示例演示如何使用 tidymodels 家族与 modeltime 拓展包构建基于机器学习的时间序列模型。
# 核心流程包括：数据准备、特征工程、模型训练、模型比较以及未来预测。

# 加载所需的 R 包 -----------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidymodels)   # tidymodels 核心框架（recipes、parsnip、workflows 等）
  library(modeltime)    # tidymodels 风格的时间序列建模扩展
  library(timetk)       # 时间序列特征工程与可视化工具
})

# 1. 获取示例数据 -------------------------------------------------------------------
# modeltime 自带的 m750 数据集是一个典型的月度时间序列。
# 变量 description 表示产品线，date 为时间索引，value 为观测值。
data(m750, package = "modeltime")

# 过滤出一条产品线的数据。这里选择 "M750" 产品线。
m750_tbl <- m750 %>%
  filter(id == "M750") %>%
  select(date, value)

# 2. 划分训练集与测试集 -------------------------------------------------------------
# 使用 time_series_split 按时间顺序切分数据。
# assess = 12 表示最后 12 期（约 1 年）作为测试集，cumulative = TRUE 表示训练集逐步累计。
split <- time_series_split(m750_tbl, assess = 12, cumulative = TRUE)

# 3. 特征工程 -----------------------------------------------------------------------
# 创建 recipe（配方）以便对时间特征进行扩展。
# step_timeseries_signature 会生成丰富的时间特征，如年份、月份、季度等。
# step_normalize 用于缩放数值型特征，step_dummy 将分类变量哑变量化。
# 最后 step_rm(...) 移除 recipe 自动生成的未使用列。
rec <- recipe(value ~ date, training(split)) %>%
  step_timeseries_signature(date) %>%
  step_normalize(contains("index.num"), contains("date")) %>%
  step_dummy(contains("lbl"), one_hot = TRUE) %>%
  step_rm(matches("(.iso$|.xts$)"))

# 4. 定义模型 -----------------------------------------------------------------------
# 这里使用 XGBoost 梯度提升树模型来拟合时间序列（作为回归问题）。
# engine = "xgboost" 指定底层算法，mode = "regression" 表示回归任务。
boost_spec <- boost_tree(
  trees = 1000,
  learn_rate = 0.05,
  tree_depth = 6,
  loss_reduction = 0.01
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# 5. 构建工作流并拟合 ---------------------------------------------------------------
# workflow 将 recipe 与模型组合，确保数据预处理与模型训练一起执行。
boost_wf <- workflow() %>%
  add_model(boost_spec) %>%
  add_recipe(rec)

# 使用训练集拟合模型。
boost_fit <- boost_wf %>% fit(training(split))

# 6. 将模型注册到 modeltime 表 ------------------------------------------------------
model_tbl <- modeltime_table(
  boost_fit
)

# 7. 模型评估 -----------------------------------------------------------------------
# calibrate_and_plot() 可快速可视化预测效果。这里我们单独展示评价指标。
# modeltime_calibrate 会将模型在测试集上的预测与真实值整合，用于后续评估。
calibration_tbl <- model_tbl %>%
  modeltime_calibrate(new_data = testing(split))

# 输出评估指标（RMSE、MAE 等），帮助了解模型表现。
calibration_tbl %>%
  modeltime_accuracy()

# 8. 未来预测 -----------------------------------------------------------------------
# 使用 modeltime_forecast 对未来 12 期进行预测。
# new_data = testing(split) 可以得到测试集上的预测结果；未来预测则需提供 future frame。
future_tbl <- calibration_tbl %>%
  modeltime_forecast(
    new_data = testing(split),
    actual_data = m750_tbl
  )

# 查看预测结果前 6 行。
future_tbl %>% head()

# 9. 绘制预测图 ---------------------------------------------------------------------
# 使用 autoplot 可视化预测与实际值的对比。
future_tbl %>%
  filter(.key != "actual") %>%
  autoplot(m750_tbl, level = NULL) +
  labs(
    title = "基于 tidymodels 的时间序列预测示例",
    subtitle = "模型：XGBoost 梯度提升树",
    y = "需求量",
    x = "时间"
  ) +
  theme_minimal()

# 提示：若在交互式环境中运行，本脚本将输出模型评估结果与预测图。
# 在非交互式环境下（如 batch 执行），请确保已安装所需 R 包。
