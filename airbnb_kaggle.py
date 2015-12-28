#airbnb_kaggle

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

df_train_users = pd.read_csv('data/train_users.csv')
df_age_gender = pd.read_csv('data/age_gender_bkts.csv')
df_countries = pd.read_csv('data/countries.csv')
df_submission = pd.read_csv('data/sample_submission_NDF.csv')
df_test_users = pd.read_csv('data/test_users.csv')
df_sessions = pd.read_csv('data/sessions.csv')

print df_train_users.head()
print df_sessions.head()
# df_train_sessions = df_train_users.merge(df_sessions)

def view_all():
	print df_train_users.head()
	print '----------------------++++++++++++++++++++++-----------------+++++++++++++++++'
	print df_age_gender.head()
	print '----------------------++++++++++++++++++++++-----------------+++++++++++++++++'
	print df_countries.head()
	print '----------------------++++++++++++++++++++++-----------------+++++++++++++++++'
	print df_submission.head()
	print '----------------------++++++++++++++++++++++-----------------+++++++++++++++++'
	print df_test_users.head()
	print '----------------------++++++++++++++++++++++-----------------+++++++++++++++++'
	print df_sessions.head()

country_destination_dict = {
	'NDF': 0,
	'US': 1,
	'other': 2,
	'FR': 3,
	'IT': 4,
	'GB': 5,
	'ES': 6,
	'CA': 7,
	'DE': 8,
	'NL': 9,
	'AU': 10,
	'PT': 11}

def prep_train_users(df):
	cols_to_remove = ['id', 'date_account_created', 'timestamp_first_active', 'date_first_booking','signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_device_type', 'first_browser', 'signup_method_facebook', 'first_affiliate_tracked_marketing', 'first_affiliate_tracked_product']
	df['country_destination'] = df['country_destination'].apply(lambda x: country_destination_dict[x])
	df['date_account_created'] = pd.to_datetime(df['date_account_created'], yearfirst=True)
	df['date_first_booking'] = pd.to_datetime(df['date_first_booking'], yearfirst=True, errors='ignore')
	df[['signup_method_facebook', 'signup_method_basic', 'signup_method_google']] = pd.get_dummies(df['signup_method'])
	df.drop(['signup_method','signup_method_google'], axis=1, inplace=True)
	df[['signup_app_android', 'signup_app_moweb', 'signup_app_web', 'signup_app_ios']] = pd.get_dummies(df['signup_app'])
	df.drop(['signup_app','signup_app_android'], axis=1, inplace=True)
	df[['gender_unknown','gender_female', 'gender_male', 'gender_other']] = pd.get_dummies(df['gender'])
	df.drop(['gender','gender_other'], axis=1, inplace=True)
	df[['first_affiliate_tracked_linked','first_affiliate_tracked_local_ops', 'first_affiliate_tracked_marketing', 'first_affiliate_tracked_omg', 'first_affiliate_tracked_product', 'first_affiliate_tracked_other', 'first_affiliate_tracked_untracked']] = pd.get_dummies(df['first_affiliate_tracked'])
	df.drop(['first_affiliate_tracked','first_affiliate_tracked_local_ops'], axis=1, inplace=True)
	df['made_booking'] = pd.isnull(df['date_first_booking'])

	df.drop(cols_to_remove, axis=1, inplace=True)
	df.fillna(value=0, inplace=True)

df_train_copy = df_train_users.copy()
prep_train_users(df_train_copy)

# print df_train_copy.head()

def prep_X_y(df):
	cols = df.columns
	y = df['country_destination']
	y = y.reshape(y.shape[0], 1)
	df.drop('country_destination', axis=1, inplace=True)
	scaler = StandardScaler()
	X = df
	X = sm.add_constant(X)
	X = scaler.fit_transform(X)
	X = pd.DataFrame(X)
	X.columns = cols
	# X = np.vstack((cols,X))
	# print X.shape
	# print y.shape
	# y_label = np.array(['country_destination']).reshape(1,1)
	# print y_label.shape
	# y = np.vstack((y_label,y))
	return X, y

def sm_ols(df):
	X, y = prep_X_y(df)

	mod = sm.OLS(y, X, missing='drop')
	res = mod.fit()
	print res.summary()

# sm_ols(df_train_copy)