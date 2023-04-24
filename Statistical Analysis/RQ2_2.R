library(dplyr)
library(lsr)
library(effsize)

LSTM <- c(0.875912,0.840975,0.788972,0.8464,0.865221,0.825678,0.887537,
          0.811306,0.641277,0.634938,0.777827,0.73107,0.76446,0.773298,
          0.74727,0.723654,0.663341,0.833333,0.770854,0.738052,0.803094)

GRU <- c(0.819205,0.781046,0.782094,0.846188,0.862861,0.830256,0.788885,
         0.814871,0.631579,0.637253,0.821062,0.743573,0.772588,0.771694,
         0.747954,0.735403,0.721676,0.795225,0.777531,0.753363,0.805821)

CNN <- c(0.750195,0.776011,0.773003,0.770894,0.772194,0.708958,0.797633,
         0.618151,0.675949,0.682725,0.695906,0.701105,0.572773,0.717517,
         0.624579,0.66237,0.645539,0.736,0.668353,0.626276,0.627191)

RF <- c(0.7856,0.798982,0.571597,0.852863,0.687991,0.61671,0.664654,
             0.773179,0.797171,0.567786,0.85983,0.699214,0.617948,0.663207,
             0.687273,0.744518,0.512821,0.84667,0.648833,0.573064,0.614944)

XGBoost <- c(0.77037,0.407278,0.422025,0.689416,0.509829,0.540997,0.53286,
        0.773585,0.472995,0.460475,0.760943,0.540507,0.447368,0.579164,
        0.684609,0.744518,0.512821,0.76487,0.542016,0.465765,0.538714) 

my_data <- data.frame(
  #group = rep(c("LSTM", "GRU"), each = 21),
  #group = rep(c("LSTM", "CNN"), each = 21),
  #group = rep(c("LSTM", "XGBoost"), each = 21),
  #group = rep(c("LSTM", "RF"), each = 21),
  #group = rep(c("GRU", "CNN"), each = 21),
  #group = rep(c("GRU", "XGBoost"), each = 21),
  #group = rep(c("GRU", "RF"), each = 21),
  #group = rep(c("CNN", "XGBoost"), each = 21),
  #group = rep(c("CNN", "RF"), each = 21),
  group = rep(c("XGBoost", "RF"), each = 21),
  #F1 = c(LSTM, GRU)
  #F1 = c(LSTM, CNN)
  #F1 = c(LSTM, XGBoost)
  #F1 = c(LSTM, RF)
  #F1 = c(GRU, CNN)
  #F1 = c(GRU, XGBoost)
  #F1 = c(GRU, RF)
  #F1 = c(CNN, XGBoost)
  #F1 = c(CNN, RF)
  F1 = c(XGBoost, RF)
)
group_by(my_data, group) %>%
  summarise(
    count = n(),
    mean = mean(F1, na.rm = TRUE),
    sd = sd(F1, na.rm = TRUE)
  )

res <- wilcox.test(F1 ~ group, data = my_data, paired=TRUE)
res
#cliff.delta(LSTM, GRU)
#cliff.delta(LSTM, CNN)
#cliff.delta(LSTM, XGBoost)
#cliff.delta(LSTM, RF)
#cliff.delta(GRU, CNN)
#cliff.delta(GRU, XGBoost)
#cliff.delta(GRU, RF)
#cliff.delta(CNN, XGBoost)
#cliff.delta(CNN, RF)
cliff.delta(XGBoost, RF)