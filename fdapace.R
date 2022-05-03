# Simulated dense dataset with 200 subjects and 100 timepoints. 

library(fdapace)
library(tidyverse)
library(gridExtra)

# Set the number of subjects (N) and the number of measurements per subjects (M) 
N = 200
M = 100
set.seed(123)

# Define the continuum
s = seq(0,10,length.out = M)

# Define the mean and 2 eigencomponents
meanFunct = function(s) s + 10*exp(-(s-5)^2)
eigFunct1 = function(s) +cos(2*s*pi/10) / sqrt(5)
eigFunct2 = function(s) -sin(2*s*pi/10) / sqrt(5)

# Create FPC scores
Ksi = matrix(rnorm(N*2), ncol=2);
Ksi = apply(Ksi,2,scale)
Ksi = Ksi %*% diag(c(5,2))

# Create Y_true
yTrue = Ksi %*% t(matrix(c(eigFunct1(s),eigFunct2(s)), ncol=2)) + t(matrix(rep(meanFunct(s),N), nrow=M))

# Simulation Study (Dense)
yTrue1 = as_tibble(yTrue)
names(yTrue1) = 1:100
yTrue1$ID = 1:200
yTrue1 = yTrue1 %>% 
  gather(key=time,value=value,-ID) %>% 
  mutate(time=as.numeric(time))
p = yTrue1 %>% ggplot() + 
  geom_path(aes(x=time,y=value,group=ID),alpha=0.05)+
  ggtitle('Simulated trajectories')+
  theme_bw()
p

# Create input
# Run functional principal component analysis 

L3 = MakeFPCAInputs(IDs = rep(1:N, each=M),tVec=rep(s,N),t(yTrue))
FPCAdense = FPCA(L3$Ly, L3$Lt) 

mu = tibble(time=1:100,value=FPCAdense$mu)
p + geom_line(data=mu,aes(x=time,y=value),color='red',size=1) + 
  ggtitle("Estimated Mean Function")
set.seed(123)

# Randomly assign NA to simulated data
# Predict individual trajectories from obtained FPCA object. 

LyNA = lapply(L3$Ly,function(x)ifelse(runif(M)>0.3,x,NA))
yPred=predict(FPCAdense,LyNA, L3$Lt, K=2)
yPred=as_tibble(yPred$predCurves)
names(yPred) = 1:100
yPred$ID = 1:200
yPred = yPred %>% gather(key=time,value=value,-ID) %>% mutate(time=as.numeric(time))
ppred = yPred %>% ggplot() + geom_path(aes(x=time,y=value,group=ID),alpha=0.05)+ggtitle('Predicted trajectories')+theme_bw()
com = bind_rows(yTrue1 %>% mutate(type='True Y'),yPred %>% mutate(type='Predicted Y')) %>% filter(ID>=193) 
com %>% ggplot() + 
  geom_path(aes(x=time,y=value,color=type),alpha=0.5)+
  ggtitle('')+theme_bw()+facet_wrap(~ID,nrow=2)+
  ggtitle("predicted mean vs true mean for subjects 193 to 200")

# The sum of differences between true mean and predicted mean for all 200 subjects is 
sum(com$value[com$type=='True Y'] - com$value[com$type=='Predicted Y'])

# Diagnostic plots 
plot(FPCAdense)

# Associated standard deviation associated with each component is
round(sqrt(FPCAdense$lambda),3)

# Simulation Study (Sparse and Noisy)
ySparse = Sparsify(yTrue, s, sparsity = c(1:5))
ySparse$yNoisy = lapply(ySparse$Ly, function(x) x + 0.5*rnorm(length(x)))
FPCAsparse = FPCA(ySparse$yNoisy, ySparse$Lt, list(plot = TRUE))

# Associated standard deviation associated with each component is 
round(sqrt(FPCAsparse$lambda),3)

# Example of usage of other functions
# FPCA calculates the bandwidth utilized by each smoother using generalised cross-validation or k-fold cross-validation automatically. 
# Dense data are not smoothed by default. 
# The argument `methodMuCovEst` can be switched between smooth and cross-sectional if one wants to utilize different estimation techniques when work with dense data. \

FPCAsparseMuBW5 = FPCA(ySparse$yNoisy, ySparse$Lt, optns= list(userBwMu = 5))
par(mfrow=c(1,2))
CreatePathPlot(FPCAsparse, subset = 1:3, main = "GCV bandwidth", pch = 16)
CreatePathPlot(FPCAsparseMuBW5, subset = 1:3, main = "User-defined bandwidth", pch = 16)

# FPCA returns automatically the smallest number of components required to explain 99% of a sample's variance. Using the function `selectK` one can determine the number of relevant components according to AIC, BIC or a different Fraction-of-Variance-Explained threshold. For example: \

SelectK(FPCAsparse, criterion = 'FVE', FVEthreshold = 0.95)
SelectK(FPCAsparse, criterion = 'AIC')
SelectK(FPCAsparse, criterion = 'BIC')

# When working with functional data (usually not very sparse) the estimation of derivatives is often of interest. Using fitted.FPCA one can directly obtain numerical derivatives by defining the appropriate order p (up to 2). \

fittedCurvesP0 = fitted(FPCAsparse, derOptns=list(p = 0))
fittedCurcesP1 = fitted(FPCAsparse, derOptns=list(p = 1, kernelType = 'epan'))

# A real-world example
# Data is taken from Accelerometer Biometric Competition from Kaggle. 
# https://www.kaggle.com/c/accelerometer-biometric-competition/data
# Variables: Time, X, Y, Z, Device.
# 29.5 million observations, 387 unique devices.

d=readRDS('watch.rds')


# Diagnostic from FPCA

X = MakeFPCAInputs(d$Device,d$Time,d$X)
fpcaObj = FPCA(X$Ly, X$Lt, list(plot = TRUE, methodMuCovEst = 'smooth'))

# Path Plots

par(mfrow=c(1,2))
CreatePathPlot(fpcaObj, subset = c(1,2), K = 4, main = 'K = 2', pch = 4); grid()
CreatePathPlot(fpcaObj, subset = c(1,2), K = 6, main = 'K = 6', pch = 4) ; grid()

# Outlier Detection

par(mfrow=c(1,1))
CreateOutliersPlot(fpcaObj, optns = list(K = 6, variant = 'KDE'))

# Visualize data with functional boxplot

CreateFuncBoxPlot(fpcaObj, xlab = 'Time', ylab = 'X - movement', optns = list(K=6, variant='bagplot'))

# Rate of change of functional data

par(mfrow=c(1,2))
CreatePathPlot(fpcaObj, subset = c(1,3), K = 6, main = 'K = 6', showObs = FALSE)
grid()
CreatePathPlot(fpcaObj, subset = c(1,3), K = 6, main = 'K = 6', showObs = FALSE, 
               derOptns = list(p = 1, bw = 1.01 , kernelType = 'epan'))
grid()

d = readRDS('watch_test.rds')
X = MakeFPCAInputs(d$Device,d$Time,d$X)
yPred=predict(fpcaObj,X$Ly,X$Lt, K=6)
yPred=as_tibble(yPred$predCurves)[1:20]
names(yPred) = 1:20
yPred$Device = unique(d$Device)

Actual = tibble(Time=1:20,Score=d$X[d$Device==7])
Predicted = tibble(Time=1:20,Score=as.numeric(yPred[yPred$Device==7,1:20]))

ggplot() +
  geom_point(data=Actual,aes(Time,Score)) +
  geom_line(data=Predicted,aes(Time,Score))