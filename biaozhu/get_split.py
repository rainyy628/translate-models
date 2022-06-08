#分割中庸——待注释2，分为三份，进行重新标注
import pandas as pd
from sklearn.model_selection import train_test_split
file_name='/root/code/yaoyao/mb/biaozhu/中庸_待注释2.csv'
df=pd.read_csv(file_name)

_,df_verify=train_test_split(df,test_size=0.1,random_state=0)
df_verify,df_verify_0=train_test_split(df_verify,test_size=0.34,random_state=0)
df_verify_1,df_verify_2=train_test_split(df_verify,test_size=0.5,random_state=0)

print(len(df_verify_0))
print(len(df_verify_1))
print(len(df_verify_2))
df_verify_0.to_csv('/root/code/yaoyao/mb/biaozhu/中庸验证2_1.csv')
df_verify_1.to_csv('/root/code/yaoyao/mb/biaozhu/中庸验证2_2.csv')
df_verify_2.to_csv('/root/code/yaoyao/mb/biaozhu/中庸验证2_3.csv')

# df_1,df_other=train_test_split(df,test_size=0.66,random_state=0)
# df_2,df_3=train_test_split(df,test_size=0.5,random_state=0)

# df_1.to_csv('/root/code/yaoyao/mb/biaozhu/中庸2_1待注释.csv')
# df_2.to_csv('/root/code/yaoyao/mb/biaozhu/中庸2_2待注释.csv')
# df_3.to_csv('/root/code/yaoyao/mb/biaozhu/中庸2_3待注释.csv')
