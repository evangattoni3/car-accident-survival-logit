
# Categorical Data Analysis Project: Logistic Regression
# Outcome: Survival in a car accident (0(died)| 1(survived))
# Goal: Model survival probability and assess predictor effects
############################################################

# ---- 1. Load libraries ----
library(dplyr)
library(broom)
library(car)

# ---- 2. Data preprocessing ----
data = crash.data

head(data)


data <- data %>%
  mutate(
    Survived       = factor(Survived, levels = c(0, 1)),
    Gender         = factor(Gender),
    Helmet_Used    = factor(Helmet_Used, levels = c("No", "Yes")),
    Seatbelt_Used  = factor(Seatbelt_Used, levels = c("No", "Yes"))
  )

# Set meaningful reference levels (important for interpretation)
data$Gender <- relevel(data$Gender, ref = "Male")

# ---- 3. Exploratory categorical analysis ----
# Marginal distributions
table(data$Survived)
prop.table(table(data$Survived))

# Independence checks (chi-square tests)
chisq.test(table(data$Survived, data$Helmet_Used))
chisq.test(table(data$Survived, data$Seatbelt_Used))
chisq.test(table(data$Survived, data$Gender))

# ---- 4. Logistic regression: full model ----
model_full <- glm(
  Survived ~ Age + Gender + Speed_of_Impact + Helmet_Used + Seatbelt_Used,
  data = data,
  family = binomial(link = "logit")
)

summary(model_full)

# ---- 5. Odds ratios with confidence intervals ----
odds_ratios <- tidy(model_full, exponentiate = TRUE, conf.int = TRUE)
odds_ratios

# ---- 6. Reduced model (variable selection / parsimony) ----
model_reduced <- glm(
  Survived ~ Speed_of_Impact + Helmet_Used + Seatbelt_Used,
  data = data,
  family = binomial(link = "logit")
)

summary(model_reduced)

# ---- 7. Model comparison ----
anova(model_reduced, model_full, test = "Chisq")
AIC(model_full, model_reduced)

# ---- 8. Multicollinearity check ----
vif(model_full)

# ---- 9. Goodness-of-fit (deviance-based) ----
with(model_full, cbind(
  Deviance = deviance,
  DF = df.residual,
  Deviance_per_DF = deviance / df.residual
))

###############
