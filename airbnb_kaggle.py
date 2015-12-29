#airbnb_kaggle

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df_train_users = pd.read_csv('data/train_users.csv')
df_age_gender = pd.read_csv('data/age_gender_bkts.csv')
df_countries = pd.read_csv('data/countries.csv')
df_submission = pd.read_csv('data/sample_submission_NDF.csv')
df_test_users = pd.read_csv('data/test_users.csv')
df_sessions = pd.read_csv('data/sessions.csv')


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
	'NDF': 1,
	'US': 2,
	'other': 3,
	'FR': 4,
	'IT': 5,
	'GB': 6,
	'ES': 7,
	'CA': 8,
	'DE': 9,
	'NL': 10,
	'AU': 11,
	'PT': 12,
	'test': -1}

def train_test_merge(df_train, df_test):
	df_test['country_destination'] = 'test'
	df_test['source'] = 'test'
	df_train['source'] = 'train'
	df_all_users = pd.concat([df_train, df_test], axis=0)
	return df_all_users

def merge_users_sessions(df_users, df_sessions, how='left'):
	#merge users and sessions tables
	return df_users.merge(df_sessions, how=how, left_on='id', right_on='user_id')

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
	df.fillna(value=-1, inplace=True)

def prep_users_sessions(df):
	users_cols_to_remove = ['id', 'timestamp_first_active', 'date_first_booking','signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_device_type', 'first_browser', 'signup_method_facebook', 'first_affiliate_tracked_marketing', 'first_affiliate_tracked_product', 'action_type_click', 'action_type_data']
	sessions_cols_to_remove = ['user_id', 'action', 'action_detail', 'device_type']
	
	#change target variable to integers for model input
	df['country_destination'] = df['country_destination'].apply(lambda x: country_destination_dict[x])

	#split date_account_created into year, month, day and dayofweek integers
	df['date_account_created'] = pd.to_datetime(df['date_account_created'], yearfirst=True)
	df['dac_yr'] = df['date_account_created'].apply(lambda x: int(x.year))
	df['dac_mo'] = df['date_account_created'].apply(lambda x: int(x.month))
	df['dac_day'] = df['date_account_created'].apply(lambda x: int(x.day))
	df['dac_dow'] = df['date_account_created'].apply(lambda x: int(x.dayofweek))
	df.drop(['date_account_created'], axis=1, inplace=True)

	# df['date_first_booking'] = pd.to_datetime(df['date_first_booking'], yearfirst=True, errors='ignore')
	# df[['signup_method_facebook', 'signup_method_basic', 'signup_method_google']] = pd.get_dummies(df['signup_method'])
	# df.drop(['signup_method','signup_method_google'], axis=1, inplace=True)
	# df[['signup_app_android', 'signup_app_moweb', 'signup_app_web', 'signup_app_ios']] = pd.get_dummies(df['signup_app'])
	# df.drop(['signup_app','signup_app_android'], axis=1, inplace=True)
	# df[['gender_unknown','gender_female', 'gender_male', 'gender_other']] = pd.get_dummies(df['gender'])
	# df.drop(['gender','gender_other'], axis=1, inplace=True)
	# df[['first_affiliate_tracked_linked','first_affiliate_tracked_local_ops', 'first_affiliate_tracked_marketing', 'first_affiliate_tracked_omg', 'first_affiliate_tracked_product', 'first_affiliate_tracked_other', 'first_affiliate_tracked_untracked']] = pd.get_dummies(df['first_affiliate_tracked'])
	# df.drop(['first_affiliate_tracked','first_affiliate_tracked_local_ops'], axis=1, inplace=True)
	# df['made_booking'] = pd.isnull(df['date_first_booking'])

	# df[['action_type_unknown','action_type_booking_request', 'action_type_booking_response', 'action_type_click', 'action_type_data', 'action_type_message_post', 'action_type_partner_callback', 'action_type_submit', 'action_type_view']] = pd.get_dummies(df['action_type'])
	# df.drop(['action_type','action_type_booking_response'], axis=1, inplace=True)

	# df.drop(users_cols_to_remove + sessions_cols_to_remove, axis=1, inplace=True)
	# df.fillna(value=-1, inplace=True)




df_all_users = train_test_merge(df_train_users, df_test_users)
df_all_users_sessions = merge_users_sessions(df_all_users, df_sessions, how='left')
prep_users_sessions(df_all_users_sessions)


print df_all_users_sessions.head()
print df_all_users_sessions.tail()


# df_merge_train = prep_merge_sessions(df_train_copy, df_sessions_copy)
# df_merge_test = prep_merge_sessions(df_test_copy, df_sessions_copy, test=True)

def prep_X_y(df, values=False):
	if not values:
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
		return X, y
	else:
		y = df.pop('country_destination').values
		X = sm.add_constant(df.values)
		X = scaler.fit_transform(X)
		return X, y

def sm_ols(df):
	X, y = prep_X_y(df)
	mod = sm.OLS(y, X)
	res = mod.fit()
	print res.summary()

def sk_logreg(df):
	X_train, y_train = prep_X_y(df, values=True)
	X_test, y_test = prep_X_y(df_merge_test, values=True)
	lr = LogisticRegression()
	lr.fit(X, y)
	y_pred = lr.predict(X_test)
	print lr.score(y_test, y_pred)

# sm_ols(df_merge_train)

# sk_logreg(df_merge_train)


