plotCD <- function (results.matrix, alpha=0.05, cex=0.75, ...) {
  
  opar <- par(mai = c(0,0,0,0))
  on.exit(par(opar))
  
  k <- dim(results.matrix)[2]
  N <- dim(results.matrix)[1]
  cd <- getNemenyiCD(alpha=alpha, num.alg=k, num.problems=N)
  
  mean.rank <- sort(colMeans(rankMatrix(results.matrix, ...)))
  
  # Separate the algorithms in left and right parts
  lp <- round(k/2)
  left.algs <- mean.rank[1:lp]
  right.algs <- mean.rank[(lp+1):k]  
  max.rows <- ceiling(k/2)
  
  # Basic dimensions and definitions
  char.size    <- 0.001  # Character size
  line.spacing <- 0.25   # Line spacing for the algorithm name
  m            <- floor(min(mean.rank))
  M            <- ceiling(max(mean.rank))
  max.char     <- max(sapply(colnames(results.matrix), FUN = nchar))  # Longest length of a label
  text.width   <- (max.char + 4) * char.size
  w            <- (M-m) + 2 * text.width
  h.up         <- 2.5 * line.spacing  # The upper part is fixed. Extra space is for the CD
  h.down       <- (max.rows + 2.25) * line.spacing # The lower part depends on the no. of algorithms. 
  # The 2 extra spaces are for the lines that join algorithms
  tick.h       <- 0.25 * line.spacing
  
  label.displacement <- 0.25    # Displacement of the label with respect to the axis
  line.displacement  <- 0.025  # Displacement for the lines that join algorithms
  
  # Background of the plot
  plot(0, 0, type="n", xlim=c(m - w / (M - m), M + w / (M - m)), 
       ylim=c(-h.down, h.up), xaxt="n", yaxt="n", xlab= "", ylab="", bty="n")
  
  # Draw the axis
  lines (c(m,M), c(0,0))
  dk <- sapply(m:M, 
               FUN=function(x) {
                 lines(c(x,x), c(0, tick.h))
                 text(x, 3*tick.h, labels=x, cex=cex)
               })
  
  # Draw the critical difference
  lines(c(m, m + cd), c(1.75 * line.spacing, 1.75 * line.spacing))
  text(m + cd / 2, 2.25 * line.spacing, "CD", cex=cex)
  lines(c(m, m), c(1.75 * line.spacing - tick.h / 4, 
                   1.75 * line.spacing + tick.h / 4))
  lines(c(m + cd, m + cd), c(1.75 * line.spacing - tick.h / 4, 
                             1.75 * line.spacing + tick.h / 4))
  
  # Left part, labels
  dk <- sapply (1:length(left.algs), 
                FUN=function(x) {
                  line.h <- -line.spacing * (x + 2)
                  text(x=m - label.displacement, y=line.h, 
                       labels=names(left.algs)[x], cex=cex, adj=1)
                  lines(c(m - label.displacement*0.75, left.algs[x]), c(line.h, line.h),lwd=2)
                  lines(c(left.algs[x], left.algs[x]), c(line.h, 0),lwd=2)
                })
  
  # Right part, labels
  dk <- sapply (1:length(right.algs), 
                FUN=function(x) {
                  line.h <- -line.spacing * (x + 2)
                  text(x=M + label.displacement, y=line.h, 
                       labels=names(right.algs)[x], cex=cex, adj=0)
                  lines(c(M + label.displacement*0.75, right.algs[x]), c(line.h, line.h),lwd=2)
                  lines(c(right.algs[x], right.algs[x]), c(line.h, 0),lwd=2)
                })
  
  # Draw the lines to join algorithms
  getInterval <- function (x) {
    from <- mean.rank[x]
    diff <- mean.rank - from
    ls <- which(diff > 0 & diff < cd)
    if (length(ls) > 0) {
      c(from, mean.rank[max(ls)])
    }
  }
  
  intervals <- mapply (1:k, FUN=getInterval)
  aux <- do.call(rbind, intervals)
  if(NROW(aux) > 0) {
    # With this strategy, there can be intervals included into bigger ones
    # We remove them in a sequential way
    to.join <- aux[1,]
    if(nrow(aux) > 1) {  
      for (r in 2:nrow(aux)) {
        if (aux[r - 1, 2] < aux[r, 2]) {
          to.join <- rbind(to.join, aux[r, ])
        }
      }
    }
    
    row <- c(1)
    # Determine each line in which row will be displayed
    if (!is.matrix(to.join)) {  # To avoid treating vector separately
      to.join <- t(as.matrix(to.join))
    }
    nlines <- dim(to.join)[1]
    
    for(r in 1:nlines) {
      id <- which(to.join[r, 1] > to.join[, 2])
      if(length(id) == 0) {
        row <- c(row, tail(row, 1) + 1)
      } else {
        row <- c(row, min(row[id]))
      }
    }
    
    step <- max(row) / 2
    
    # Draw the line
    dk <- sapply (1:nlines, 
                  FUN = function(x) {
                    y <- -line.spacing * (0.5 + row[x] / step)
                    lines(c(to.join[x, 1] - line.displacement, 
                            to.join[x, 2] + line.displacement), 
                          c(y, y), lwd=3,col="gray")
                  })
  }
}

rankMatrix <- function(data, decreasing=TRUE, ...){
  # The rank function is based on an increasing ordering. In case we need to
  # get the rank of the descreasing ordering, just rank -x instead of x
  if (decreasing){
    f <- function(x){
      rank(-x, ties.method="average")
    }
  } else {
    f <- function(x){
      rank(x, ties.method="average")
    }
  }
  
  rankings <- t(apply (data, MARGIN=1, FUN=f))
  colnames(rankings) <- colnames(data)
  rownames(rankings) <- rownames(data)
  return(rankings)
}


getNemenyiCD <- function (alpha = 0.05, num.alg, num.problems) {
  # Auxiliar function to compute the critical difference for Nemenyi test
  # Args:
  #   alpha:        Alpha for the test
  #   num.alg:      Number of algorithms tested
  #   num.problems: Number of problems where the algorithms have been tested
  #
  # Returns:
  #   Corresponding critical difference
  #
  df <- num.alg * (num.problems - 1)
  qa <- qtukey(p = 1 - alpha,
               nmeans = num.alg,
               df = df) / sqrt(2)
  cd <- qa * sqrt((num.alg * (num.alg + 1)) / (6 * num.problems))
  return(cd)
}

# 精度
Word2Vec_XGBoost <- c(0.936,0.9738,0.7709,0.9887,0.9765,0.8925,0.9082)
Word2Vec_RF <- c(0.9665,0.9584,0.8929,0.9938,0.9753,0.8936,0.9588)
Word2Vec_LSTM <- c(0.9708,0.8803,0.8449,0.9353,0.9062,0.8088,0.9252)
Word2Vec_GRU <- c(0.8921,0.8608,0.7657,0.9557,0.912,0.8536,0.73)
Word2Vec_CNN <- c(0.8019,0.8672,0.8024,0.8306,0.8598,0.8116,0.9172)
fastText_XGBoost <- c(0.9699,0.9514,0.7717,0.9765,0.9600,0.8854,0.9136)
fastText_RF <- c(0.9550,0.9572,0.9313,0.9928,0.9765,0.9047,0.9604)
fastText_LSTM <- c(0.8879,0.7802,0.6138,0.9004,0.7361,0.8271,0.8266)
fastText_GRU <- c(0.8832,0.6912,0.6237,0.9311,0.8,0.8385,0.8108)
fastText_CNN <- c(0.8957,0.7719,0.7272,0.9284,0.7699,0.7138,0.8255)
codeBert_XGBoost <- c(0.9378,0.9245,0.9079,0.9763,0.9449,0.8797,0.9083)
codeBert_RF <- c(0.9356,0.9245,0.9079,0.9663,0.9604,0.8867,0.9243)
codeBert_LSTM <- c(0.8568,0.7944,0.6072,0.9226,0.8852,0.7846,0.8607)
codeBert_GRU <- c(0.8908,0.793,0.6863,0.9665,0.8781,0.8364,0.8135)
codeBert_CNN <- c(0.7827,0.7604,0.6947,0.9221,0.7986,0.7429,0.7211)

# 召回�?
# Word2Vec_XGBoost <- c(0.6545,0.2575,0.2905,0.5292,0.3449,0.3881,0.377)
# Word2Vec_RF <- c(0.6617,0.6820,0.4203,0.7469,0.5314,0.4708,0.5086)
# Word2Vec_LSTM <- c(0.7978,0.805,0.7399,0.7729,0.8277,0.8432,0.8527)
# Word2Vec_GRU <- c(0.7572,0.7147,0.7991,0.7582,0.8187,0.8081,0.8581)
# Word2Vec_CNN <- c(0.7046,0.7021,0.7456,0.7191,0.7007,0.6293,0.7056)
# fastText_XGBoost <- c(0.6434,0.3147,0.3281,0.6233,0.3761,0.2993,0.4239)
# fastText_RF <- c(0.6495,0.6830,0.4084,0.7583,0.5446,0.4692,0.5065)
# fastText_LSTM <- c(0.7468,0.5443,0.6575,0.6846,0.726,0.7106,0.7264)
# fastText_GRU <- c(0.7563,0.5813,0.6513,0.7343,0.6945,0.7162,0.7361)
# fastText_CNN <- c(0.4718,0.6012,0.6433,0.5565,0.6435,0.4782,0.6345)
# codeBert_XGBoost <- c(0.5391,0.6232,0.3573,0.6287,0.3800,0.3167,0.3829)
# codeBert_RF <- c(0.5431,0.6232,0.3573,0.7534,0.4899,0.4233,0.4607)
# codeBert_LSTM <- c(0.6625,0.6644,0.7307,0.7598,0.6827,0.6967,0.7527)
# codeBert_GRU <- c(0.6446,0.6855,0.7608,0.6755,0.6976,0.6853,0.7982)
# codeBert_CNN <- c(0.5196,0.5867,0.6028,0.6123,0.5745,0.5412,0.5549)

# F1
# Word2Vec_XGBoost <- c(0.77037,0.407278,0.422025,0.689416,0.509829,0.540997,0.53286)
# Word2Vec_RF <- c(0.7856,0.798982,0.571597,0.852863,0.687991,0.61671,0.664654)
# Word2Vec_LSTM <- c(0.875912,0.840975,0.788972,0.8464,0.865221,0.825678,0.887537)
# Word2Vec_GRU <- c(0.819205,0.781046,0.782094,0.846188,0.862861,0.830256,0.788885)
# Word2Vec_CNN <- c(0.750195,0.776011,0.773003,0.770894,0.772194,0.708958,0.797633)
# fastText_XGBoost <- c(0.773585,0.472995,0.460475,0.760943,0.540507,0.447368,0.579164)
# fastText_RF <- c(0.773179,0.797171,0.567786,0.85983,0.699214,0.617948,0.663207)
# fastText_LSTM <- c(0.811306,0.641277,0.634938,0.777827,0.73107,0.76446,0.773298)
# fastText_GRU <- c(0.814871,0.631579,0.637253,0.821062,0.743573,0.772588,0.771694)
# fastText_CNN <- c(0.618151,0.675949,0.682725,0.695906,0.701105,0.572773,0.717517)
# codeBert_XGBoost <- c(0.684609,0.744518,0.512821,0.76487,0.542016,0.465765,0.538714)
# codeBert_RF <- c(0.687273,0.744518,0.512821,0.84667,0.648833,0.573064,0.614944)
# codeBert_LSTM <- c(0.74727,0.723654,0.663341,0.833333,0.770854,0.738052,0.803094)
# codeBert_GRU <- c(0.747954,0.735403,0.721676,0.795225,0.777531,0.753363,0.805821)
# codeBert_CNN <- c(0.624579,0.66237,0.645539,0.736,0.668353,0.626276,0.627191)

mat <- cbind(Word2Vec_XGBoost, Word2Vec_RF, Word2Vec_LSTM, Word2Vec_GRU, Word2Vec_CNN,
             fastText_XGBoost, fastText_RF, fastText_LSTM, fastText_GRU, fastText_CNN,
             codeBert_XGBoost, codeBert_RF, codeBert_LSTM, codeBert_GRU, codeBert_CNN)
rownames(mat) <- c('xss','xsrf','sql','remote code execution','path disclosure','open redirect','command injection')
mat
# write.csv(mat,"Friedman_F1.csv")

result <- friedman.test(mat)
result$statistic
result$p.value

plotCD(results.matrix = mat, alpha = 0.05)