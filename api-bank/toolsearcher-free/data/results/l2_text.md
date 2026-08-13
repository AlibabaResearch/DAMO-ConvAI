# API-Bank Grading Report

- **Endpoint**: `https://176.108.242.226:443/v1/chat/completions`
- **Model**: `Qwen/Qwen3.6-35B-A3B-FP8`
- **Mode**: `text`
- **Level**: 2
- **Datapoints graded**: 49

## Aggregate

| metric | value |
|---|---:|
| total gold tool-calls | 64 |
| tool-name exact | 34 (53.1%) |
| args exact (full-call match) | 24 (37.5%) |
| args partial (≥50% keys match) | 46.9% |
| mean args fraction | 0.431 |
| no-call datapoints | 3 |

## Per-datapoint verdicts

| id | gold | pred | name | args-exact | args-frac |
|---:|---|---|:--:|:--:|---:|
| 0 | QueryHistoryToday({"date": "10-06"}) | QueryHistoryToday(...) | yes | yes | 1.00 |
| 1 | GetUserToken({"username": "user2", "password": "user2) | GetUserToken(...) | yes | yes | 1.00 |
| 2 | QueryStock({"stock_code": "MSFT", "date": "2022-02-) | QueryStock(...) | yes | yes | 1.00 |
| 3 | GetUserToken({"username": "JohnDoe", "password": "pas) | DeleteAgenda(...) | no | no | 0.00 |
| 3 | DeleteAgenda({"token": "a9s8d7f6g5h4j3k2l1", "content) | DeleteAgenda(...) | yes | no | 0.50 |
| 4 | Calculator({"formula": "8*8"}) | Calculator(...) | yes | no | 0.00 |
| 5 | GetToday({}) | GetToday(...) | yes | yes | 1.00 |
| 6 | GetUserToken({"username": "user4", "password": "user4) | AddAlarm(...) | no | no | 0.00 |
| 6 | AddAlarm({"token": "q9w8e7r6t5y4u3i2o1", "time": ) | AddAlarm(...) | yes | yes | 1.00 |
| 7 | GetUserToken({"username": "newuser", "password": "new) | AddAgenda(...) | no | no | 0.00 |
| 7 | AddAgenda({"token": "l9k8j7h6g5f4d3s2a1", "content) | AddAgenda(...) | yes | no | 0.75 |
| 8 | ModifyRegistration({"appointment_id": "34567890", "new_appo) | ModifyRegistration(...) | yes | yes | 1.00 |
| 9 | GetUserToken({"username": "user2", "password": "user2) | AddReminder(...) | no | no | 0.00 |
| 9 | QueryReminder({"token": "o9i8u7y6t5r4e3w2q1", "content) | AddReminder(...) | no | no | 0.00 |
| 10 | QueryHealthData({"user_id": "F24681", "start_time": "202) | QueryHealthData(...) | yes | no | 0.33 |
| 11 | GetUserToken({"username": "admin", "password": "admin) | AddReminder(...) | no | no | 0.00 |
| 11 | DeleteReminder({"token": "m9n8b7v6c5x4z3a2s1", "content) | AddReminder(...) | no | no | 0.00 |
| 12 | QueryHistoryToday({"date": "10-06"}) | QueryHistoryToday(...) | yes | yes | 1.00 |
| 13 | GetUserToken({"username": "JohnDoe", "password": "pas) | DeleteMeeting(...) | no | no | 0.00 |
| 13 | DeleteAgenda({"token": "a9s8d7f6g5h4j3k2l1", "content) | None(...) | no | no | 0.00 |
| 14 | Calculator({"formula": "25*3+7/3"}) | Calculator(...) | yes | yes | 1.00 |
| 15 | GetUserToken({"username": "user4", "password": "user4) | QueryAlarm(...) | no | no | 0.00 |
| 15 | QueryAlarm({"token": "q9w8e7r6t5y4u3i2o1", "time": ) | QueryAlarm(...) | yes | no | 0.50 |
| 16 | GetUserToken({"username": "newuser", "password": "new) | AddMeeting(...) | no | no | 0.00 |
| 16 | AddAgenda({"token": "l9k8j7h6g5f4d3s2a1", "content) | None(...) | no | no | 0.00 |
| 17 | GetUserToken({"username": "newuser", "password": "new) | AddReminder(...) | no | no | 0.00 |
| 17 | AddReminder({"token": "l9k8j7h6g5f4d3s2a1", "content) | AddReminder(...) | yes | yes | 1.00 |
| 18 | GetUserToken({"username": "user4", "password": "user4) | AddAlarm(...) | no | no | 0.00 |
| 18 | AddAlarm({"token": "q9w8e7r6t5y4u3i2o1", "time": ) | AddAlarm(...) | yes | yes | 1.00 |
| 19 | EmergencyKnowledge({"symptom": "nausea"}) | SymptomSearch(...) | no | no | 0.00 |
| 20 | QueryRegistration({"patient_name": "John Doe", "date": "20) | QueryRegistration(...) | yes | yes | 1.00 |
| 21 | SymptomSearch({"symptom": "rash"}) | SymptomSearch(...) | yes | no | 0.00 |
| 22 | BookHotel({"hotel_name": "Hilton", "check_in_time") | BookHotel(...) | yes | no | 0.50 |
| 23 | GetUserToken({"username": "user2", "password": "user2) | GetToday(...) | no | no | 0.00 |
| 23 | DeleteAccount({"token": "o9i8u7y6t5r4e3w2q1"}) | DeleteAccount(...) | yes | yes | 1.00 |
| 25 | GetUserToken({"username": "JohnDoe", "password": "pas) | GetToday(...) | no | no | 0.00 |
| 26 | GetUserToken({"username": "user4", "password": "user4) | ModifyMeeting(...) | no | no | 0.00 |
| 27 | GetUserToken({"username": "JaneSmith", "password": "p) | DeleteAlarm(...) | no | no | 0.00 |
| 27 | DeleteAlarm({"token": "o8i7u6y5t4r3e2w1q0", "time": ) | DeleteAlarm(...) | yes | no | 0.50 |
| 28 | GetUserToken({"username": "admin", "password": "admin) | DeleteMeeting(...) | no | no | 0.00 |
| 29 | SymptomSearch({"symptom": "rash"}) | SymptomSearch(...) | yes | yes | 1.00 |
| 30 | GetUserToken({"username": "foo", "password": "bar"}) | GetUserToken(...) | yes | yes | 1.00 |
| 30 | QueryBalance({"token": "z9x8c7v6b5n4m3q2w1"}) | QueryBalance(...) | yes | yes | 1.00 |
| 31 | GetUserToken({"username": "testuser", "password": "te) | QueryMeeting(...) | no | no | 0.00 |
| 32 | QueryRegistration({"patient_name": "John Doe", "date": "20) | QueryRegistration(...) | yes | no | 0.50 |
| 33 | AppointmentRegistration({"patient_name": "John Doe", "date": "20) | GetToday(...) | no | no | 0.00 |
| 34 | GetUserToken({"username": "user2", "password": "user2) | DeleteAccount(...) | no | no | 0.00 |
| 34 | DeleteAccount({"token": "o9i8u7y6t5r4e3w2q1"}) | DeleteAccount(...) | yes | yes | 1.00 |
| 35 | GetUserToken({"username": "JohnDoe", "password": "pas) | QueryReminder(...) | no | no | 0.00 |
| 35 | ModifyReminder({"token": "a9s8d7f6g5h4j3k2l1", "content) | QueryReminder(...) | no | no | 0.00 |
| 36 | SymptomSearch({"symptom": "rash"}) | SymptomSearch(...) | yes | yes | 1.00 |
| 37 | BookHotel({"hotel_name": "Hilton Hotel", "check_in) | BookHotel(...) | yes | yes | 1.00 |
| 40 | GetUserToken({"username": "JohnDoe", "password": "pas) | AddMeeting(...) | no | no | 0.00 |
| 41 | SymptomSearch({"symptom": "rash"}) | SymptomSearch(...) | yes | yes | 1.00 |
| 42 | BookHotel({"hotel_name": "Grand Hotel", "check_in_) | BookHotel(...) | yes | yes | 1.00 |
| 43 | GetUserToken({"username": "user2", "password": "user2) | GetUserToken(...) | yes | no | 0.00 |
| 43 | DeleteAccount({"token": "o9i8u7y6t5r4e3w2q1"}) | DeleteAccount(...) | yes | yes | 1.00 |
| 44 | GetUserToken({"username": "admin", "password": "admin) | DeleteMeeting(...) | no | no | 0.00 |
| 45 | GetUserToken({"username": "foo", "password": "bar"}) | QueryAgenda(...) | no | no | 0.00 |
| 45 | ModifyAgenda({"token": "z9x8c7v6b5n4m3q2w1", "content) | QueryAgenda(...) | no | no | 0.00 |
| 46 | GetUserToken({"username": "testuser", "password": "te) | QueryMeeting(...) | no | no | 0.00 |
| 47 | GetUserToken({"username": "foo", "password": "bar"}) | GetUserToken(...) | yes | yes | 1.00 |
| 47 | QueryBalance({"token": "z9x8c7v6b5n4m3q2w1"}) | QueryBalance(...) | yes | yes | 1.00 |
| 48 | SymptomSearch({"symptom": "rash"}) | SymptomSearch(...) | yes | yes | 1.00 |