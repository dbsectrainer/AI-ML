# R Programming for Machine Learning

This section covers R programming fundamentals and advanced concepts for statistical computing and machine learning applications.

## Core R Concepts

### 1. R Basics
- Data types and structures
- Vectors and matrices
- Data frames and lists
- Functions and control structures
- Package management

### 2. Data Manipulation
```r
# Using dplyr
library(dplyr)

# Data transformation
mtcars %>%
  group_by(cyl) %>%
  summarise(
    mean_mpg = mean(mpg),
    sd_mpg = sd(mpg)
  ) %>%
  arrange(desc(mean_mpg))

# Using data.table
library(data.table)
dt <- as.data.table(mtcars)
dt[, .(mean_mpg = mean(mpg), sd_mpg = sd(mpg)), by = cyl]
```

### 3. Data Visualization
```r
# Base R plotting
plot(mtcars$wt, mtcars$mpg)
abline(lm(mpg ~ wt, data = mtcars))

# ggplot2
library(ggplot2)
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_minimal() +
  labs(title = "Weight vs MPG by Cylinders",
       x = "Weight",
       y = "Miles per Gallon")
```

## Statistical Analysis

### 1. Descriptive Statistics
```r
# Summary statistics
summary(mtcars)

# Advanced statistics
library(psych)
describe(mtcars)

# Correlation analysis
cor(mtcars[, c("mpg", "wt", "hp")])
library(corrplot)
corrplot(cor(mtcars))
```

### 2. Statistical Tests
```r
# t-test
t.test(mpg ~ am, data = mtcars)

# ANOVA
model <- aov(mpg ~ factor(cyl), data = mtcars)
summary(model)

# Chi-square test
chisq.test(table(mtcars$cyl, mtcars$am))
```

## Machine Learning in R

### 1. Linear Models
```r
# Linear regression
lm_model <- lm(mpg ~ wt + hp, data = mtcars)
summary(lm_model)

# Generalized linear models
glm_model <- glm(am ~ mpg + wt, 
                 family = binomial(link = "logit"), 
                 data = mtcars)
```

### 2. Advanced ML Models
```r
# Random Forest
library(randomForest)
rf_model <- randomForest(mpg ~ ., data = mtcars)
importance(rf_model)

# Support Vector Machines
library(e1071)
svm_model <- svm(mpg ~ ., data = mtcars)
```

### 3. Model Evaluation
```r
# Cross-validation
library(caret)
ctrl <- trainControl(method = "cv", number = 10)
model <- train(mpg ~ ., data = mtcars, 
              method = "rf", 
              trControl = ctrl)

# ROC curves
library(pROC)
roc_curve <- roc(response, predictions)
plot(roc_curve)
```

## Data Processing

### 1. Data Cleaning
```r
# Handling missing values
library(tidyr)
df %>%
  drop_na() %>%
  replace_na(list(x = mean(x, na.rm = TRUE)))

# Outlier detection
library(outliers)
scores <- scores(mtcars$mpg, type = "z")
```

### 2. Feature Engineering
```r
# Creating new features
mtcars %>%
  mutate(
    power_to_weight = hp/wt,
    efficiency = mpg/disp
  )

# One-hot encoding
library(caret)
dummy <- dummyVars(" ~ .", data = mtcars)
encoded <- predict(dummy, mtcars)
```

## Performance Optimization

### 1. Parallel Processing
```r
library(parallel)
library(doParallel)

# Setup parallel backend
cores <- detectCores()
cl <- makeCluster(cores-1)
registerDoParallel(cl)

# Parallel operations
foreach(i = 1:1000) %dopar% {
  # Computations
}

stopCluster(cl)
```

### 2. Memory Management
```r
# Memory profiling
library(pryr)
mem_used()
object.size(mtcars)

# Garbage collection
gc()
```

## Best Practices

### 1. Code Organization
- Use RStudio projects
- Follow tidyverse style guide
- Document code with roxygen2
- Use version control (git)

### 2. Package Development
```r
library(devtools)
create_package("mypackage")
use_testthat()
document()
```

## Resources

### Books
- "R for Data Science" by Hadley Wickham
- "Advanced R" by Hadley Wickham
- "Statistical Learning with R" by James et al.

### Online Resources
- RStudio Education
- Tidyverse documentation
- R-bloggers
- Stack Overflow R community

### Tools
- RStudio IDE
- R Markdown
- Shiny for interactive applications
- packrat/renv for dependency management

## Assessment Questions

1. Compare dplyr and data.table approaches
2. Explain different cross-validation methods in R
3. How do you optimize R code for performance?
4. Describe best practices for R package development

## Projects

1. Build a statistical analysis pipeline
2. Create a machine learning model comparison tool
3. Develop an R package for custom ML algorithms
4. Create interactive visualizations with Shiny
