# API-Bank Grading Report

- **Endpoint**: `https://176.108.242.226:443/v1/chat/completions`
- **Model**: `Qwen/Qwen3.6-35B-A3B-FP8`
- **Mode**: `text`
- **Level**: 3 (batch)
- **Datapoints graded**: 121

## Aggregate

| metric | value |
|---|---:|
| total gold tool-calls | 121 |
| tool-name exact | 105 (86.8%) |
| args exact (full-call match) | 85 (70.2%) |
| args partial (≥50% keys match) | 78.5% |
| mean args fraction | 0.761 |
| no-call datapoints | 5 |

## Per-datapoint verdicts

| id | gold | pred | name | args-exact | args-frac |
|---:|---|---|:--:|:--:|---:|
| 1 | QueryMeeting({"user_name": "John"}) | QueryMeeting(...) | yes | no | 0.00 |
| 3 | EmailReminder({"content": "Meeting with the client wil) | EmailReminder(...) | yes | no | 0.75 |
| 4 | EmailReminder({"content": "Meeting about the new proje) | EmailReminder(...) | yes | yes | 1.00 |
| 6 | QueryMeeting({"user_name": "Mary"}) | QueryMeeting(...) | yes | no | 0.00 |
| 8 | EmailReminder({"content": "Meeting with the client wil) | EmailReminder(...) | yes | no | 0.75 |
| 10 | QueryMeeting({"user_name": "Peter"}) | QueryMeeting(...) | yes | no | 0.00 |
| 12 | EmailReminder({"content": "Meeting with the client wil) | EmailReminder(...) | yes | no | 0.75 |
| 14 | QueryMeeting({"user_name": "Tom"}) | QueryMeeting(...) | yes | no | 0.00 |
| 16 | EmailReminder({"content": "Meeting will be held at Roo) | EmailReminder(...) | yes | no | 0.75 |
| 18 | QueryMeeting({"user_name": "Jerry"}) | QueryMeeting(...) | yes | no | 0.00 |
| 20 | EmailReminder({"content": "Meeting will be held at Roo) | EmailReminder(...) | yes | no | 0.75 |
| 22 | GetWeatherForCoordinates({"latitude": "40.7128", "longitude": "74) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 24 | ClothingRecommendation({"temperature": "10", "humidity": "0.5",) | ClothingRecommendation(...) | yes | yes | 1.00 |
| 26 | GetWeatherForCoordinates({"latitude": "37.7749", "longitude": "12) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 28 | ClothingRecommendation({"temperature": "20", "humidity": "0.8",) | ClothingRecommendation(...) | yes | yes | 1.00 |
| 30 | GetWeatherForCoordinates({"latitude": "51.5074", "longitude": "0.) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 32 | ClothingRecommendation({"temperature": "5", "humidity": "0.9", ) | ClothingRecommendation(...) | yes | yes | 1.00 |
| 34 | GetWeatherForCoordinates({"latitude": "48.8566", "longitude": "2.) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 36 | ClothingRecommendation({"temperature": "15", "humidity": "0.7",) | ClothingRecommendation(...) | yes | no | 0.67 |
| 38 | GetWeatherForCoordinates({"latitude": "35.6762", "longitude": "13) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 40 | ClothingRecommendation({"temperature": "25", "humidity": "0.6",) | ClothingRecommendation(...) | yes | yes | 1.00 |
| 42 | OrganizationMembers({"organization": "Alibaba"}) | OrganizationMembers(...) | yes | yes | 1.00 |
| 44 | TravelStatus({"member_name": "John"}) | TravelStatus(...) | yes | yes | 1.00 |
| 45 | TravelStatus({"member_name": "Mary"}) | TravelStatus(...) | yes | yes | 1.00 |
| 46 | TravelStatus({"member_name": "Peter"}) | TravelStatus(...) | yes | yes | 1.00 |
| 50 | OrganizationMembers({"organization": "Tencent"}) | OrganizationMembers(...) | yes | yes | 1.00 |
| 52 | TravelStatus({"member_name": "Tom"}) | TravelStatus(...) | yes | yes | 1.00 |
| 53 | TravelStatus({"member_name": "Jerry"}) | TravelStatus(...) | yes | yes | 1.00 |
| 57 | OrganizationMembers({"organization": "Baidu"}) | OrganizationMembers(...) | yes | yes | 1.00 |
| 59 | TravelStatus({"member_name": "Jack"}) | TravelStatus(...) | yes | yes | 1.00 |
| 60 | TravelStatus({"member_name": "Rose"}) | TravelStatus(...) | yes | yes | 1.00 |
| 64 | OrganizationMembers({"organization": "ByteDance"}) | OrganizationMembers(...) | yes | yes | 1.00 |
| 66 | TravelStatus({"member_name": "Bob"}) | TravelStatus(...) | yes | yes | 1.00 |
| 67 | TravelStatus({"member_name": "Alice"}) | TravelStatus(...) | yes | yes | 1.00 |
| 71 | OrganizationMembers({"organization": "JD"}) | OrganizationMembers(...) | yes | yes | 1.00 |
| 73 | TravelStatus({"member_name": "Mike"}) | TravelStatus(...) | yes | yes | 1.00 |
| 74 | TravelStatus({"member_name": "Jane"}) | None(...) | no | no | 0.00 |
| 78 | Geocoding({"address": "New York City"}) | NearbyRestaurants(...) | no | no | 0.00 |
| 80 | NearbyRestaurants({"latitude": "40.7128", "longitude": "74) | NearbyRestaurants(...) | yes | no | 0.67 |
| 82 | Geocoding({"address": "San Francisco"}) | NearbyRestaurants(...) | no | no | 0.00 |
| 84 | NearbyRestaurants({"latitude": "37.7749", "longitude": "12) | NearbyRestaurants(...) | yes | no | 0.67 |
| 86 | Geocoding({"address": "London"}) | NearbyRestaurants(...) | no | no | 0.00 |
| 88 | NearbyRestaurants({"latitude": "51.5074", "longitude": "0.) | NearbyRestaurants(...) | yes | yes | 1.00 |
| 90 | Geocoding({"address": "Paris"}) | NearbyRestaurants(...) | no | no | 0.00 |
| 92 | NearbyRestaurants({"latitude": "48.8566", "longitude": "2.) | NearbyRestaurants(...) | yes | yes | 1.00 |
| 94 | Geocoding({"address": "Tokyo"}) | NearbyRestaurants(...) | no | no | 0.00 |
| 96 | NearbyRestaurants({"latitude": "35.6762", "longitude": "13) | NearbyRestaurants(...) | yes | yes | 1.00 |
| 98 | UserMoviePreferences({"user_name": "John"}) | UserMoviePreferences(...) | yes | yes | 1.00 |
| 100 | UserWatchedMovies({"user_name": "John"}) | UserWatchedMovies(...) | yes | yes | 1.00 |
| 104 | UserMoviePreferences({"user_name": "Mary"}) | UserMoviePreferences(...) | yes | yes | 1.00 |
| ... | _71 more rows truncated — see full JSON_ | | | | |