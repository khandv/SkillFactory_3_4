import pandas as pd
import warnings

warnings.filterwarnings("ignore")

events_df = pd.read_csv('7_4_Events.csv')
purshases_df = pd.read_csv('7_4_purchase.csv')

# данные для пользователей, зарегистрировавшихся в 2018 году
cond = (events_df['start_time'] >= '2018-01-01') & (events_df['start_time'] < '2019-01-01') & \
       (events_df['event_type'] == 'registration')
registered = events_df[cond]['user_id'].to_list()  # список пользователей, зарег. в 2018

events_reg_2018 = events_df[events_df['user_id'].isin(registered)]
purshases_reg_2018 = purshases_df[purshases_df['user_id'].isin(registered)]

registered_users_count = events_reg_2018[events_reg_2018['event_type'] == 'registration']['user_id'].nunique()
tutorial_start_users_count = events_reg_2018[events_reg_2018['event_type'] == 'tutorial_start']['user_id'].nunique()
percent_tutorial_start_users = tutorial_start_users_count / registered_users_count
print('Процент пользователей, начавших обучение (от общего числа зарегистрировавшихся): '
      '{:.2%}'.format(percent_tutorial_start_users))

tutorial_finish_users_count = events_reg_2018[events_reg_2018['event_type'] == 'tutorial_finish']['user_id'].nunique()
tutorial_completion_rate = tutorial_finish_users_count / tutorial_start_users_count
print('Процент пользователей, завершивших обучение: {:.2%}'.format(tutorial_completion_rate))

level_choice_users_count = events_reg_2018[events_reg_2018['event_type'] == 'level_choice']['user_id'].nunique()
percent_level_choice_users = level_choice_users_count / registered_users_count
print('Процент пользователей, выбравших уровень сложности тренировок (от общего числа зарегистрировавшихся): '
      '{:.2%}'.format(percent_level_choice_users))

training_choice_users_count = events_reg_2018[events_reg_2018['event_type'] == 'pack_choice']['user_id'].nunique()
percent_training_choice_users = training_choice_users_count / level_choice_users_count
print('Процент пользователей, выбравших набор бесплатных вопросов '
      '(от числа пользователей, которые выбрали уровень сложности): {:.2%}'.format(percent_training_choice_users))

paying_users_count = purshases_reg_2018['user_id'].nunique()
percent_of_paying_users = paying_users_count / training_choice_users_count
print('Процент пользователей, которые оплатили вопросы (от числа пользователей, которые выбрали тренировки): '
      '{:.2%}'.format(percent_of_paying_users))
purchase_rate = paying_users_count / registered_users_count
print('Процент пользователей, которые оплатили вопросы(от числа зарегистрировавшихся пользователей): '
      '{:.2%}'.format(purchase_rate))

purshases_reg_2018['event_type'] = 'purchase'
events_reg_2018 = events_reg_2018.rename(columns={'id': 'event_id'})
purshases_reg_2018 = purshases_reg_2018.rename(columns={'id': 'purchase_id'})
total_events_df = pd.concat([events_reg_2018, purshases_reg_2018], sort=False)
total_events_df = total_events_df.reset_index(drop=True).sort_values('start_time')

total_events_df['event_type'] = total_events_df['event_type'].astype(str)

'''Теперь можно для каждого пользователя создать список, который будет содержать во временной последовательности 
все события, совершаемые данным пользователем. Для этого воспользуемся методом groupby по столбцу event_type и 
применим аргегирующую функцию apply(list). Таким образом, мы сгруппируем строки по пользователю, 
а затем объединим в списки содержимое столбца event_type по каждому пользователю.'''
user_path_df = total_events_df.groupby(['user_id'])['event_type'].apply(list).reset_index()

'''Прежде чем мы сможем оценить популярные пути пользователей, преобразуем список событий в строку event_path. 
Эта операция нужна для оптимизации скорости объединения, так как иначе pandas может делать подсчёт слишком долго.'''
user_path_df['event_path'] = user_path_df['event_type'].apply(lambda x: '>'.join(x))

'''Теперь можно сгруппировать датафрейм по столбцу event_path, подсчитав число пользователей:'''
user_paths = user_path_df.groupby(['event_path'])['user_id'].nunique().sort_values(ascending=False)

'''какие ещё последовательности содержат в себе оплату:'''
# print(user_paths[user_paths.index.str.contains('purchase')].head(10))

'''выделим отдельный датафрейм registration_df, который будет содержать только события с event_type = registration.'''
registration_df = total_events_df[total_events_df['event_type'] == 'registration']

'''Подсчет кол-ва по колоке и вывод среднего'''
# print(registration_df['user_id'].value_counts().mean())

'''оставим в датафрейме registration_df только те данные, которые нужны для наших вычислений — столбец user_id 
с идентификатором пользователя и столбец start_time со временем регистрации. Также переименуем столбец start_time в 
столбец registration_time для понятности.'''
registration_df = registration_df[['user_id', 'start_time']].rename(columns={'start_time': 'registration_time'})

'''Выделим отдельный датафрейм tutorial_start_df, который будет содержать только события с event_type = 
                                                                                    tutorial_start (начало обучения).'''
tutorial_start_df = total_events_df[total_events_df['event_type'] == 'tutorial_start']

'''Создадим датафрейм tutorial_start_df_wo_duplicates, где будет присутствовать только первое обучение. 
Для этого сначала отсортируем датафрейм по start_time, чтобы сначала шли более ранние события начала обучения, 
а затем удалим дубликаты по столбцу user_id. Таким образом, для каждого user_id останется только первое событие типа 
tutorial_start.'''
tutorial_start_df_wo_duplicates = tutorial_start_df.sort_values('start_time').drop_duplicates('user_id')
'''оставим только такие столбцы, которые пригодятся нам в дальнейшем. Это столбцы user_id, start_time, tutorial_id. 
Также переименуем колонку start_time в tutorial_start_time:'''
tutorial_start_df_wo_duplicates = tutorial_start_df_wo_duplicates[['user_id', 'tutorial_id', 'start_time']]. \
    rename(columns={'start_time': 'tutorial_start_time'})

'''Объединztv между собой данные двух получившихся датафреймов: registration_df и tutorial_start_df_wo_duplicates.'''
merged_df = registration_df.merge(tutorial_start_df_wo_duplicates, on='user_id', how='inner')

'''Сделаем столбец timedelta, в котором посчитаем разницу между временем начала обучения tutorial_start_time и 
временем регистрации registration_time. Предварительно приведем данные столцов в формат дата-время:'''
merged_df['tutorial_start_time'] = pd.to_datetime(merged_df['tutorial_start_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df['registration_time'] = pd.to_datetime(merged_df['registration_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df['timedelta'] = merged_df['tutorial_start_time'] - merged_df['registration_time']

'''Сформируем датафрейм tutorial_finish_df, который содержит события окончания обучения:'''
tutorial_finish_df = total_events_df[total_events_df['event_type'] == 'tutorial_finish']
'''Сформируем список с идентификаторами первых обучений.'''
first_tutorial_ids = tutorial_start_df_wo_duplicates['tutorial_id'].unique()
'''Отфильтруем датафрейм tutorial_finish_df, оставив в нём только события для таких обучений, 
которые были первыми для пользователя.'''
tutorial_finish_df = tutorial_finish_df[tutorial_finish_df['tutorial_id'].isin(first_tutorial_ids)]
'''Переименуем колонку времени'''
tutorial_finish_df = tutorial_finish_df[['user_id', 'start_time']].rename(
    columns={'start_time': 'tutorial_finish_time'})

'''Объединяем 2 датафрейма начала и конца обучения, ищем разницу во времени'''
merged_df_2 = tutorial_start_df_wo_duplicates.merge(tutorial_finish_df, on='user_id', how='inner')
merged_df_2['tutorial_start_time'] = pd.to_datetime(merged_df_2['tutorial_start_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_2['tutorial_finish_time'] = pd.to_datetime(merged_df_2['tutorial_finish_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_2['timedelta'] = merged_df_2['tutorial_finish_time'] - merged_df_2['tutorial_start_time']

'''По такому же алгоритму рассчитаем время между выбором уровня сложности и регистрацией:'''
level_choice_df = total_events_df[total_events_df['event_type'] == 'level_choice']
level_choice_df = level_choice_df[['user_id', 'start_time']].rename(columns={'start_time': 'level_choice_time'})
merged_df_3 = registration_df.merge(level_choice_df, on='user_id', how='inner')
merged_df_3['registration_time'] = pd.to_datetime(merged_df_3['registration_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_3['level_choice_time'] = pd.to_datetime(merged_df_3['level_choice_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_3['timedelta'] = merged_df_3['level_choice_time'] - merged_df_3['registration_time']

'''Время между событием выбора уровня сложности тренировки до события выбора набора бесплатных тренировок'''
pack_choice_df = total_events_df[total_events_df['event_type'] == 'pack_choice']
pack_choice_df = pack_choice_df[['user_id', 'start_time']].rename(columns={'start_time': 'pack_choice_time'})
merged_df_4 = level_choice_df.merge(pack_choice_df, on='user_id', how='inner')
merged_df_4['pack_choice_time'] = pd.to_datetime(merged_df_4['pack_choice_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_4['level_choice_time'] = pd.to_datetime(merged_df_4['level_choice_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_4['timedelta'] = merged_df_4['pack_choice_time'] - merged_df_4['level_choice_time']

'''Время между событием выбора набора бесплатных тренировок и первой оплатой'''
purchase_df = total_events_df[total_events_df['event_type'] == 'purchase']
purchase_df = purchase_df[['user_id', 'event_datetime', 'amount']].rename(columns={'event_datetime': 'purshase_time'})
merged_df_5 = pack_choice_df.merge(purchase_df, on='user_id', how='inner')
merged_df_5['pack_choice_time'] = pd.to_datetime(merged_df_5['pack_choice_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_5['purshase_time'] = pd.to_datetime(merged_df_5['purshase_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_5['timedelta'] = merged_df_5['purshase_time'] - merged_df_5['pack_choice_time']
# print(merged_df_5['timedelta'].describe())

'''Пользователи, которые прошли обучение хотя бы раз'''
users_with_finished_tutorial = total_events_df[total_events_df['event_type'] == 'tutorial_finish']['user_id'].unique()
'''Пользователи, которые начали обучение, но не прошли его ни разу. 
Это будут пользователи, у которых есть событие tutorial_start, но нет события tutorial_finish.
Решим через множества'''
users_with_started_tutorial = total_events_df[total_events_df['event_type'] == 'tutorial_start']['user_id'].unique()
set_users_with_started_tutorial = set(users_with_started_tutorial)
set_users_not_finished_but_started_tutorial = set_users_with_started_tutorial.difference(
    set(users_with_finished_tutorial))

'''Кто ни разу не проходил обучение. У таких пользователей отсутствует событие tutorial_start. 
Поэтому мы можем просто взять и убрать из множества всех пользователей множество set_users_with_started_tutorial:'''
all_users = total_events_df['user_id'].unique()
set_all_users = set(all_users)
set_users_not_started_tutorial = set_all_users.difference(set_users_with_started_tutorial)

'''Данные по оплатам пользователей, которые завершили обучение:'''
purchase_df_1 = purchase_df[purchase_df['user_id'].isin(users_with_finished_tutorial)]
'''Процент оплативших пользователей в этой группе:'''
percent_of_purchase_1 = purchase_df_1['user_id'].nunique() / len(users_with_finished_tutorial)
print('Процент пользователей, которые оплатили тренировки (от числа пользователей, завершивших обучение): '
      '{:.2%}'.format(percent_of_purchase_1))
amount_mean_1 = purchase_df_1['amount'].mean()
print('Средний платеж завершивших обучение: {:.2f}'.format(amount_mean_1))

'''Оплаты пользователей, которые начали обучение, но не закончили. И какой процент таких пользователей оплачивает 
пакеты вопросов, от общего числа пользователей:'''
purchase_df_2 = purchase_df[purchase_df['user_id'].isin(set_users_not_finished_but_started_tutorial)]
percent_of_purchase_2 = purchase_df_2['user_id'].nunique() / len(set_users_not_finished_but_started_tutorial)
print('Процент пользователей, которые оплатили тренировки (от числа пользователей, начавших обучение, '
      'но не завершивших): {:.2%}'.format(percent_of_purchase_2))
amount_mean_2 = purchase_df_2['amount'].mean()
print('Cредний платеж начавших, но не завершивших обучение: {:.2f}'.format(amount_mean_2))

'''Оплаты пользователей, которые ни разу не начинали обучение'''
purchase_df_3 = purchase_df[purchase_df['user_id'].isin(set_users_not_started_tutorial)]
percent_of_purchase_3 = purchase_df_3['user_id'].nunique() / len(set_users_not_started_tutorial)
print('Процент пользователей, которые оплатили тренировки (от числа пользователей, ни разу не начинавших обучение): '
      '{:.2%}'.format(percent_of_purchase_3))
amount_mean_3 = purchase_df_3['amount'].mean()
print('Cредний платеж ни разу не начинавших обучение: {:.2f}'.format(amount_mean_3))
print(120 * '=')

'''Уровни сложности'''
levels = list(total_events_df['selected_level'].unique())
levels.pop(0)

'''Легкий уровень'''
user_id_easy_level = total_events_df[total_events_df['selected_level'] == 'easy']['user_id'].unique()
easy_level_df_purchase = purchase_df[purchase_df['user_id'].isin(user_id_easy_level)]
percent_of_easy_level = easy_level_df_purchase['user_id'].nunique() / len(user_id_easy_level)
print('Процент оплативших выбравших легкий уровень сложности: {:.2%}'.format(percent_of_easy_level))
'''Считаем среднее время для легкого уровня сложности'''
easy_level_df = total_events_df[total_events_df['selected_level'] == 'easy']
merged_df_easy = easy_level_df.merge(purchase_df, on='user_id', how='inner')
merged_df_easy['start_time'] = pd.to_datetime(merged_df_easy['start_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_easy['purshase_time'] = pd.to_datetime(merged_df_easy['purshase_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_easy['timedelta'] = merged_df_easy['purshase_time'] - merged_df_easy['start_time']
print("Среднее время между выбором легкого уровня сложности и оплатой: ", merged_df_easy['timedelta'].mean())
print(120 * '-')

'''Средний уровень'''
user_id_medium_level = total_events_df[total_events_df['selected_level'] == 'medium']['user_id'].unique()
medium_level_df_purchase = purchase_df[purchase_df['user_id'].isin(user_id_medium_level)]
percent_of_medium_level = medium_level_df_purchase['user_id'].nunique() / len(user_id_medium_level)
print('Процент оплативших выбравших средний уровень сложности: {:.2%}'.format(percent_of_medium_level))
'''Считаем среднее время для среднего уровня сложности'''
medium_level_df = total_events_df[total_events_df['selected_level'] == 'medium']
merged_df_medium = medium_level_df.merge(purchase_df, on='user_id', how='inner')
merged_df_medium['start_time'] = pd.to_datetime(merged_df_medium['start_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_medium['purshase_time'] = pd.to_datetime(merged_df_medium['purshase_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_medium['timedelta'] = merged_df_medium['purshase_time'] - merged_df_medium['start_time']
print("Среднее время между выбором среднего уровня сложности и оплатой: ", merged_df_medium['timedelta'].mean())
print(120 * '-')

'''Тяжелый уровень сложности'''
user_id_hard_level = total_events_df[total_events_df['selected_level'] == 'hard']['user_id'].unique()
hard_level_df_purchase = purchase_df[purchase_df['user_id'].isin(user_id_hard_level)]
percent_of_hard_level = hard_level_df_purchase['user_id'].nunique() / len(user_id_hard_level)
print('Процент оплативших выбравших тяжелый уровень сложности: {:.2%}'.format(percent_of_hard_level))
'''Считаем среднее время для тяжелого уровня сложности'''
hard_level_df = total_events_df[total_events_df['selected_level'] == 'hard']
merged_df_hard = hard_level_df.merge(purchase_df, on='user_id', how='inner')
merged_df_hard['start_time'] = pd.to_datetime(merged_df_hard['start_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_hard['purshase_time'] = pd.to_datetime(merged_df_hard['purshase_time'], format='%Y-%m-%dT%H:%M:%S')
merged_df_hard['timedelta'] = merged_df_hard['purshase_time'] - merged_df_hard['start_time']
print("Среднее время между выбором тяжелого уровня сложности и оплатой: ", merged_df_hard['timedelta'].mean())
