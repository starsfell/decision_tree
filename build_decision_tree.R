rm(list = ls())

setwd("C:/Users/xintong.yan/Desktop")
data <- read.table("sample_data.txt", header =TRUE)

# 计算IV值的公式
IV_Dn <- function(a){
  IV_D <- -a*log2(a)
  return(IV_D)
}


# 计算ENT值的公式
Ent <- function(a, b){
  Ent_ab <- -(a*log2(a) + b*log2(b))
  return(Ent_ab)
}

# 计算Gini系数的公式
Gini_Dn <- function(a, b) {
  Gini_ab = 1 - (a^2 + b^2)
}



# Ent值与Gain值
ENT_data = Ent(9/17, 8/17)    # 总样本的Ent值


ENT_SXR_D1 = Ent(1/9, 8/9)     # 双休日为“否”的Ent值
ENT_SXR_D2 = Ent(7/8, 1/8)     # 双休日为“是”的Ent值
Gain_SXR = 0.998-((9/17)*ENT_SXR_D1+(8/17)*ENT_SXR_D2) # 双休日的gain值 


ENT_JB_D1 = Ent(6/9, 3/9)      # 加班为“否”的Ent值
ENT_JB_D2 = Ent(2/8, 6/8)      # 加班为“是”的Ent值
Gain_JB = 0.998-((9/17)*ENT_JB_D1+(8/17)*ENT_JB_D2) # 加班的gain值 


ENT_NPY_D1 = Ent(6/10,4/10)      # 女朋友为“ok”的Ent值
ENT_NPY_D2 = Ent(2/7,5/7)        # 女朋友为“不ok”的Ent值
Gain_NPY = 0.998-((10/17)*ENT_NPY_D1 +(7/17)*ENT_NPY_D2) # 女朋友的gain值 


ENT_TQ_D1 = Ent(3/5,2/5)      # 天气为“晴朗”的Ent值
ENT_TQ_D2 = Ent(4/7,3/7)      # 天气为“多云”的Ent值
ENT_TQ_D3 = Ent(1/5,4/5)      # 天气为“下雨”的Ent值
Gain_TQ = 0.998-((5/17)*ENT_TQ_D1 +(7/17)*ENT_TQ_D2+(5/17)*ENT_TQ_D3) # 天气的gain值 
# IV值
IV_BH = IV_Dn(1/17)*17                # 编号的IV值
IV_SXR = IV_Dn(8/17)+ IV_Dn(9/17)     # 双休日的IV值
IV_JB = IV_Dn(9/17)+ IV_Dn(8/17)      # 加班的IV值
IV_NPY = IV_Dn(10/17)+ IV_Dn(7/17)    # 女朋友的IV值
IV_TQ = IV_Dn(5/17)+ IV_Dn(5/17) + IV_Dn(7/17) # 天气的IV值
# 增益率
gain_ratio_SXR = Gain_SXR/IV_SXR
gain_ratio_JB = Gain_JB/IV_JB
gain_ratio_NPY = Gain_NPY/IV_NPY
gain_ratio_TQ = Gain_TQ/IV_TQ



# Gini系数


Gini_data = Gini_Dn(8/17,9/17)   # 总样本的Gini系数

Gini_SXR_D1 = Gini_Dn(1/9, 8/9)     # 双休日为“否”的Gini系数
Gini_SXR_D2 = Gini_Dn(7/8, 1/8)     # 双休日为“是”的Gini系数
Gini_SXR = (9/17)*Gini_SXR_D1+(8/17)*Gini_SXR_D2 # 双休日的Gini系数 


Gini_JB_D1 = Gini_Dn(6/9, 3/9)      # 加班为“否”的Gini系数
Gini_JB_D2 = Gini_Dn(2/8, 6/8)      # 加班为“是”的Gini系数
Gini_JB = (9/17)*Gini_JB_D1+(8/17)*Gini_JB_D2 # 加班的Gini系数


Gini_NPY_D1 = Gini_Dn(6/10,4/10)      # 女朋友为“ok”的Gini系数
Gini_NPY_D2 = Gini_Dn(2/7,5/7)        # 女朋友为“不ok”的Gini系数
Gini_NPY = (10/17)*Gini_NPY_D1 +(7/17)*Gini_NPY_D2 # 女朋友的Gini系数


Gini_TQ_D1 = Gini_Dn(3/5,2/5)      # 天气为“晴朗”的Gini系数
Gini_TQ_D2 = Gini_Dn(4/7,3/7)      # 天气为“多云”的Gini系数
Gini_TQ_D3 = Gini_Dn(1/5,4/5)      # 天气为“下雨”的Gini系数
Gini_TQ = (5/17)*Gini_TQ_D1 +(7/17)*Gini_TQ_D2+(5/17)*Gini_TQ_D3 # 天气的Gini系数 
