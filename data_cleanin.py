# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:54:37 2021

@author: airbox
"""

import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

#salary parsing

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)

df = df[df['Salary Estimate'] != "-1"]
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x:x.replace('K', '').replace('$',''))

min_hr  = minus_kd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary + df.max_salary)/2
#Company name text only

df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis= 1)

#state field

df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
#df['job_city'] = df['Location'].apply(lambda x: x.split(',')[0])

df['same_state'] = df.apply(lambda x: 1 if x['Location']==x['Headquarters'] else 0, axis=1)

#age of company 

df['age'] = df['Founded'].apply(lambda x: 2020 - int(x) if x >0 else x) 

#parsing of job description

#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
#r
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r-studio' in x.lower() or 'r studio' in x.lower() else 0)

#spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

#aws

df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

#excel

df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)


#tensor

df['tensor_yn'] = df['Job Description'].apply(lambda x: 1 if 'tensor' in x.lower() else 0)

#sql

df['sql_yn'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)

df_out = df.drop(['Unnamed: 0'], axis=1)

df_out.to_csv('salary_data_cleaned.csv', index = False)

pd.read_csv('salary_data_cleaned.csv')
