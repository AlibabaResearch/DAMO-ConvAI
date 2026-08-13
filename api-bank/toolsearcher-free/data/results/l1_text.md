# API-Bank Grading Report

- **Endpoint**: `https://176.108.242.226:443/v1/chat/completions`
- **Model**: `Qwen/Qwen3.6-35B-A3B-FP8`
- **Mode**: `text`
- **Level**: 1
- **Datapoints graded**: 368

## Aggregate

| metric | value |
|---|---:|
| total gold tool-calls | 368 |
| tool-name exact | 336 (91.3%) |
| args exact (full-call match) | 288 (78.3%) |
| args partial (≥50% keys match) | 87.8% |
| mean args fraction | 0.846 |
| no-call datapoints | 0 |

## Per-datapoint verdicts

| id | gold | pred | name | args-exact | args-frac |
|---:|---|---|:--:|:--:|---:|
| 0 | ModifyRegistration({"appointment_id": "34567890", "new_appo) | ModifyRegistration(...) | yes | yes | 1.00 |
| 1 | QueryHealthData({"user_id": "J46801", "start_time": "202) | QueryHealthData(...) | yes | no | 0.67 |
| 2 | CancelRegistration({"appointment_id": "90123456"}) | CancelRegistration(...) | yes | yes | 1.00 |
| 3 | Calculator({"formula": "(5+6)*3"}) | Calculator(...) | yes | yes | 1.00 |
| 4 | QueryScene({"name": "Morning Routine"}) | QueryScene(...) | yes | yes | 1.00 |
| 5 | EmergencyKnowledge({"symptom": "shortness of breath"}) | EmergencyKnowledge(...) | yes | yes | 1.00 |
| 6 | ModifyRegistration({"appointment_id": "90123456", "new_appo) | ModifyRegistration(...) | yes | yes | 1.00 |
| 7 | SymptomSearch({"symptom": "fatigue"}) | SymptomSearch(...) | yes | no | 0.00 |
| 8 | SymptomSearch({"symptom": "fatigue"}) | Wiki(...) | no | no | 0.00 |
| 9 | AppointmentRegistration({"patient_name": "Emily Smith", "date": ) | AppointmentRegistration(...) | yes | yes | 1.00 |
| 10 | ModifyRegistration({"appointment_id": "12345678", "new_appo) | ModifyRegistration(...) | yes | yes | 1.00 |
| 11 | EmergencyKnowledge({"symptom": "fatigue"}) | SymptomSearch(...) | no | no | 0.00 |
| 12 | EmergencyKnowledge({"symptom": "fatigue"}) | SymptomSearch(...) | no | no | 0.00 |
| 13 | GetUserToken({"username": "foo", "password": "bar"}) | GetUserToken(...) | yes | yes | 1.00 |
| 14 | DeleteAccount({"token": "z9x8c7v6b5n4m3q2w1"}) | DeleteAccount(...) | yes | yes | 1.00 |
| 15 | GetUserToken({"username": "JaneSmith", "password": "p) | GetUserToken(...) | yes | yes | 1.00 |
| 16 | QueryAlarm({"token": "o8i7u6y5t4r3e2w1q0", "time": ) | QueryAlarm(...) | yes | yes | 1.00 |
| 17 | GetUserToken({"username": "JohnDoe", "password": "pas) | GetUserToken(...) | yes | yes | 1.00 |
| 18 | GetUserToken({"username": "JohnDoe", "password": "pas) | AddAgenda(...) | no | no | 0.00 |
| 19 | AddAgenda({"token": "a9s8d7f6g5h4j3k2l1", "content) | AddAgenda(...) | yes | no | 0.75 |
| 20 | AddAgenda({"token": "a9s8d7f6g5h4j3k2l1", "content) | AddAgenda(...) | yes | no | 0.75 |
| 21 | QueryStock({"stock_code": "AMZN", "date": "2022-03-) | QueryStock(...) | yes | yes | 1.00 |
| 22 | OpenBankAccount({"account": "user4", "password": "user4p) | OpenBankAccount(...) | yes | yes | 1.00 |
| 23 | GetUserToken({"username": "user4", "password": "user4) | QueryBalance(...) | no | no | 0.00 |
| 24 | QueryBalance({"token": "q9w8e7r6t5y4u3i2o1"}) | QueryBalance(...) | yes | yes | 1.00 |
| 25 | QueryBalance({"token": "q9w8e7r6t5y4u3i2o1"}) | QueryBalance(...) | yes | yes | 1.00 |
| 26 | QueryStock({"stock_code": "SQ", "date": "2022-03-15) | QueryStock(...) | yes | yes | 1.00 |
| 27 | GetUserToken({"username": "user1", "password": "user1) | AddReminder(...) | no | no | 0.00 |
| 28 | AddReminder({"token": "n9m8k7j6h5g4f3d2s1a0", "conte) | AddReminder(...) | yes | yes | 1.00 |
| 29 | AddReminder({"token": "n9m8k7j6h5g4f3d2s1a0", "conte) | AddReminder(...) | yes | yes | 1.00 |
| 30 | CancelRegistration({"appointment_id": "78901234"}) | CancelRegistration(...) | yes | yes | 1.00 |
| 31 | QueryRegistration({"patient_name": "Jane Smith", "date": ") | QueryRegistration(...) | yes | no | 0.50 |
| 32 | ForgotPassword({"status": "Forgot Password", "username") | ForgotPassword(...) | yes | yes | 1.00 |
| 33 | ForgotPassword({"status": "Verification Code", "verific) | ForgotPassword(...) | yes | no | 1.00 |
| 34 | GetUserToken({"username": "foo", "password": "newpass) | GetUserToken(...) | yes | yes | 1.00 |
| 35 | DeleteAccount({"token": "z9x8c7v6b5n4m3q2w1"}) | DeleteAccount(...) | yes | yes | 1.00 |
| 36 | RegisterUser({"username": "foo", "password": "bar", ") | RegisterUser(...) | yes | yes | 1.00 |
| 37 | GetUserToken({"username": "user1", "password": "user1) | GetUserToken(...) | yes | yes | 1.00 |
| 38 | ModifyPassword({"token": "n9m8k7j6h5g4f3d2s1a0", "old_p) | ModifyPassword(...) | yes | yes | 1.00 |
| 39 | GetUserToken({"username": "JohnDoe", "password": "pas) | GetUserToken(...) | yes | yes | 1.00 |
| 40 | QueryReminder({"token": "a9s8d7f6g5h4j3k2l1", "content) | QueryReminder(...) | yes | no | 0.67 |
| 41 | ModifyRegistration({"appointment_id": "90123456", "new_appo) | ModifyRegistration(...) | yes | yes | 1.00 |
| 42 | ForgotPassword({"status": "Forgot Password", "username") | ForgotPassword(...) | yes | yes | 1.00 |
| 43 | ForgotPassword({"status": "Verification Code", "verific) | ForgotPassword(...) | yes | yes | 1.00 |
| 44 | GetUserToken({"username": "user1", "password": "user1) | GetUserToken(...) | yes | no | 0.50 |
| 45 | DeleteAccount({"token": "n9m8k7j6h5g4f3d2s1a0"}) | DeleteAccount(...) | yes | yes | 1.00 |
| 46 | GetUserToken({"username": "user1", "password": "user1) | GetUserToken(...) | yes | yes | 1.00 |
| 47 | AddReminder({"token": "n9m8k7j6h5g4f3d2s1a0", "conte) | AddReminder(...) | yes | yes | 1.00 |
| 48 | GetUserToken({"username": "foo", "password": "bar"}) | AddAgenda(...) | no | no | 0.00 |
| 49 | AddAgenda({"token": "z9x8c7v6b5n4m3q2w1", "content) | AddAgenda(...) | yes | yes | 1.00 |
| ... | _318 more rows truncated — see full JSON_ | | | | |