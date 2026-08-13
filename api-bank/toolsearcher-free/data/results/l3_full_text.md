# API-Bank Grading Report

- **Endpoint**: `https://176.108.242.226:443/v1/chat/completions`
- **Model**: `Qwen/Qwen3.6-35B-A3B-FP8`
- **Mode**: `text`
- **Level**: 3 (full)
- **Datapoints graded**: 50

## Aggregate

| metric | value |
|---|---:|
| total gold tool-calls | 131 |
| tool-name exact | 87 (66.4%) |
| args exact (full-call match) | 82 (62.6%) |
| args partial (≥50% keys match) | 62.6% |
| mean args fraction | 0.630 |
| no-call datapoints | 0 |

## Per-datapoint verdicts

| id | gold | pred | name | args-exact | args-frac |
|---:|---|---|:--:|:--:|---:|
| 0 | QueryMeeting({"user_name": "John"}) | EmailReminder(...) | no | no | 0.00 |
| 0 | EmailReminder({"content": "Meeting with the client wil) | EmailReminder(...) | yes | yes | 1.00 |
| 0 | EmailReminder({"content": "Meeting about the new proje) | EmailReminder(...) | yes | yes | 1.00 |
| 1 | QueryMeeting({"user_name": "Mary"}) | EmailReminder(...) | no | no | 0.00 |
| 1 | EmailReminder({"content": "Meeting with the client wil) | EmailReminder(...) | yes | no | 0.25 |
| 2 | QueryMeeting({"user_name": "Peter"}) | EmailReminder(...) | no | no | 0.00 |
| 2 | EmailReminder({"content": "Meeting with the client wil) | EmailReminder(...) | yes | no | 0.25 |
| 3 | QueryMeeting({"user_name": "Tom"}) | None(...) | no | no | 0.00 |
| 3 | EmailReminder({"content": "Meeting will be held at Roo) | EmailReminder(...) | yes | yes | 1.00 |
| 4 | QueryMeeting({"user_name": "Jerry"}) | None(...) | no | no | 0.00 |
| 4 | EmailReminder({"content": "Meeting will be held at Roo) | EmailReminder(...) | yes | yes | 1.00 |
| 5 | GetWeatherForCoordinates({"latitude": "40.7128", "longitude": "74) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 5 | ClothingRecommendation({"temperature": 10, "humidity": 0.5, "we) | ClothingRecommendation(...) | yes | yes | 1.00 |
| 6 | GetWeatherForCoordinates({"latitude": "37.7749", "longitude": "12) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 6 | ClothingRecommendation({"temperature": 20, "humidity": 0.8, "we) | ClothingRecommendation(...) | yes | yes | 1.00 |
| 7 | GetWeatherForCoordinates({"latitude": "51.5074", "longitude": "0.) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 7 | ClothingRecommendation({"temperature": 5, "humidity": 0.9, "wea) | ClothingRecommendation(...) | yes | yes | 1.00 |
| 8 | GetWeatherForCoordinates({"latitude": "48.8566", "longitude": "2.) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 8 | ClothingRecommendation({"temperature": 15, "humidity": 0.7, "we) | ClothingRecommendation(...) | yes | yes | 1.00 |
| 9 | GetWeatherForCoordinates({"latitude": "35.6762", "longitude": "13) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 9 | ClothingRecommendation({"temperature": 25, "humidity": 0.6, "we) | ClothingRecommendation(...) | yes | yes | 1.00 |
| 10 | OrganizationMembers({"organization": "Alibaba"}) | EmailReminder(...) | no | no | 0.00 |
| 10 | TravelStatus({"member_name": "John"}) | TravelStatus(...) | yes | yes | 1.00 |
| 10 | TravelStatus({"member_name": "Mary"}) | TravelStatus(...) | yes | yes | 1.00 |
| 11 | OrganizationMembers({"organization": "Tencent"}) | EmailReminder(...) | no | no | 0.00 |
| 11 | TravelStatus({"member_name": "Tom"}) | TravelStatus(...) | yes | yes | 1.00 |
| 11 | TravelStatus({"member_name": "Jerry"}) | TravelStatus(...) | yes | yes | 1.00 |
| 12 | OrganizationMembers({"organization": "Baidu"}) | EmailReminder(...) | no | no | 0.00 |
| 12 | TravelStatus({"member_name": "Jack"}) | TravelStatus(...) | yes | yes | 1.00 |
| 12 | TravelStatus({"member_name": "Rose"}) | TravelStatus(...) | yes | yes | 1.00 |
| 13 | OrganizationMembers({"organization": "ByteDance"}) | EmailReminder(...) | no | no | 0.00 |
| 13 | TravelStatus({"member_name": "Bob"}) | TravelStatus(...) | yes | yes | 1.00 |
| 13 | TravelStatus({"member_name": "Alice"}) | TravelStatus(...) | yes | yes | 1.00 |
| 14 | OrganizationMembers({"organization": "JD"}) | EmailReminder(...) | no | no | 0.00 |
| 14 | TravelStatus({"member_name": "Mike"}) | TravelStatus(...) | yes | yes | 1.00 |
| 14 | TravelStatus({"member_name": "Jane"}) | TravelStatus(...) | yes | yes | 1.00 |
| 15 | Geocoding({"address": "New York City"}) | QueryHealthData(...) | no | no | 0.00 |
| 15 | NearbyRestaurants({"latitude": "40.7128", "longitude": "74) | NearbyRestaurants(...) | yes | yes | 1.00 |
| 16 | Geocoding({"address": "San Francisco"}) | NearbyRestaurants(...) | no | no | 0.00 |
| 16 | NearbyRestaurants({"latitude": "37.7749", "longitude": "12) | NearbyRestaurants(...) | yes | yes | 1.00 |
| 17 | Geocoding({"address": "London"}) | ClothingRecommendation(...) | no | no | 0.00 |
| 17 | NearbyRestaurants({"latitude": "51.5074", "longitude": "0.) | NearbyRestaurants(...) | yes | yes | 1.00 |
| 18 | Geocoding({"address": "Paris"}) | NearbyRestaurants(...) | no | no | 0.00 |
| 18 | NearbyRestaurants({"latitude": "48.8566", "longitude": "2.) | NearbyRestaurants(...) | yes | yes | 1.00 |
| 19 | Geocoding({"address": "Tokyo"}) | NearbyRestaurants(...) | no | no | 0.00 |
| 19 | NearbyRestaurants({"latitude": "35.6762", "longitude": "13) | NearbyRestaurants(...) | yes | yes | 1.00 |
| 20 | UserMoviePreferences({"user_name": "John"}) | None(...) | no | no | 0.00 |
| 20 | UserWatchedMovies({"user_name": "John"}) | UserWatchedMovies(...) | yes | yes | 1.00 |
| 20 | MovieRecommendations({"preferences": ["Action", "Comedy", "Dr) | None(...) | no | no | 0.00 |
| 21 | UserMoviePreferences({"user_name": "Mary"}) | None(...) | no | no | 0.00 |
| 21 | UserWatchedMovies({"user_name": "Mary"}) | UserWatchedMovies(...) | yes | yes | 1.00 |
| 21 | MovieRecommendations({"preferences": ["Comedy", "Drama", "Rom) | None(...) | no | no | 0.00 |
| 22 | UserMoviePreferences({"user_name": "Peter"}) | None(...) | no | no | 0.00 |
| 22 | UserWatchedMovies({"user_name": "Peter"}) | UserWatchedMovies(...) | yes | yes | 1.00 |
| 22 | MovieRecommendations({"preferences": ["Action", "Drama", "Thr) | None(...) | no | no | 0.00 |
| 23 | UserMoviePreferences({"user_name": "Tom"}) | None(...) | no | no | 0.00 |
| 23 | UserWatchedMovies({"user_name": "Tom"}) | UserWatchedMovies(...) | yes | yes | 1.00 |
| 23 | MovieRecommendations({"preferences": ["Action", "Comedy", "Dr) | None(...) | no | no | 0.00 |
| 24 | UserMoviePreferences({"user_name": "Jerry"}) | None(...) | no | no | 0.00 |
| 24 | UserWatchedMovies({"user_name": "Jerry"}) | UserWatchedMovies(...) | yes | yes | 1.00 |
| 24 | MovieRecommendations({"preferences": ["Comedy", "Drama", "Rom) | None(...) | no | no | 0.00 |
| 25 | UserPosts({"user_id": 2}) | QueryHealthData(...) | no | no | 0.00 |
| 25 | LikeCount({"post_id": 4}) | LikeCount(...) | yes | yes | 1.00 |
| 25 | LikeCount({"post_id": 5}) | LikeCount(...) | yes | yes | 1.00 |
| 26 | UserPosts({"user_id": 1}) | QueryHealthData(...) | no | no | 0.00 |
| 26 | LikeCount({"post_id": 1}) | LikeCount(...) | yes | yes | 1.00 |
| 26 | LikeCount({"post_id": 2}) | LikeCount(...) | yes | yes | 1.00 |
| 27 | UserPosts({"user_id": 3}) | Calculator(...) | no | no | 0.00 |
| 27 | LikeCount({"post_id": 7}) | LikeCount(...) | yes | yes | 1.00 |
| 27 | LikeCount({"post_id": 8}) | LikeCount(...) | yes | yes | 1.00 |
| 28 | UserPosts({"user_id": 4}) | Calculator(...) | no | no | 0.00 |
| 28 | LikeCount({"post_id": 10}) | LikeCount(...) | yes | yes | 1.00 |
| 28 | LikeCount({"post_id": 11}) | LikeCount(...) | yes | yes | 1.00 |
| 29 | UserPosts({"user_id": 5}) | None(...) | no | no | 0.00 |
| 29 | LikeCount({"post_id": 13}) | LikeCount(...) | yes | yes | 1.00 |
| 29 | LikeCount({"post_id": 14}) | LikeCount(...) | yes | yes | 1.00 |
| 30 | Geocoding({"address": "New York City"}) | ClothingRecommendation(...) | no | no | 0.00 |
| 30 | GetWeatherForCoordinates({"latitude": "40.7128", "longitude": "74) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 31 | Geocoding({"address": "San Francisco"}) | ClothingRecommendation(...) | no | no | 0.00 |
| 31 | GetWeatherForCoordinates({"latitude": "37.7749", "longitude": "12) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 32 | Geocoding({"address": "London"}) | ClothingRecommendation(...) | no | no | 0.00 |
| 32 | GetWeatherForCoordinates({"latitude": "51.5074", "longitude": "0.) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 33 | Geocoding({"address": "Paris"}) | ClothingRecommendation(...) | no | no | 0.00 |
| 33 | GetWeatherForCoordinates({"latitude": "48.8566", "longitude": "2.) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 34 | Geocoding({"address": "Tokyo"}) | ClothingRecommendation(...) | no | no | 0.00 |
| 34 | GetWeatherForCoordinates({"latitude": "35.6762", "longitude": "13) | GetWeatherForCoordinates(...) | yes | yes | 1.00 |
| 35 | FlightSearch({"source": "New York", "destination": "S) | FlightSearch(...) | yes | no | 0.00 |
| 35 | HotelAvailability({"destination": "San Francisco", "check_) | HotelAvailability(...) | yes | yes | 1.00 |
| 36 | FlightSearch({"source": "Los Angeles", "destination":) | FlightSearch(...) | yes | yes | 1.00 |
| 36 | HotelAvailability({"destination": "San Francisco", "check_) | HotelAvailability(...) | yes | yes | 1.00 |
| 37 | FlightSearch({"source": "London", "destination": "San) | FlightSearch(...) | yes | no | 0.00 |
| 37 | HotelAvailability({"destination": "San Francisco", "check_) | HotelAvailability(...) | yes | yes | 1.00 |
| 38 | FlightSearch({"source": "New York", "destination": "L) | FlightSearch(...) | yes | yes | 1.00 |
| 38 | HotelAvailability({"destination": "London", "check_in_date) | HotelAvailability(...) | yes | yes | 1.00 |
| 39 | FlightSearch({"source": "New York", "destination": "L) | FlightSearch(...) | yes | yes | 1.00 |
| 39 | HotelAvailability({"destination": "Los Angeles", "check_in) | HotelAvailability(...) | yes | yes | 1.00 |
| 40 | GetOccupationSalary({"occupation": "Financial Analyst"}) | TaxCalculator(...) | no | no | 0.00 |
| 40 | TaxCalculator({"salary": "100000"}) | TaxCalculator(...) | yes | yes | 1.00 |
| 41 | GetOccupationSalary({"occupation": "Software Engineer"}) | Calculator(...) | no | no | 0.00 |
| 41 | TaxCalculator({"salary": "120000"}) | TaxCalculator(...) | yes | yes | 1.00 |
| 42 | GetOccupationSalary({"occupation": "Data Scientist"}) | Calculator(...) | no | no | 0.00 |
| 42 | TaxCalculator({"salary": "150000"}) | TaxCalculator(...) | yes | yes | 1.00 |
| 43 | GetOccupationSalary({"occupation": "Product Manager"}) | TaxCalculator(...) | no | no | 0.00 |
| 43 | TaxCalculator({"salary": 130000}) | TaxCalculator(...) | yes | yes | 1.00 |
| 44 | GetOccupationSalary({"occupation": "Doctor"}) | GetOccupationSalary(...) | yes | yes | 1.00 |
| 44 | TaxCalculator({"salary": "200000"}) | TaxCalculator(...) | yes | yes | 1.00 |
| 45 | AccountInfo({"username": "John", "password": "123456) | AccountInfo(...) | yes | yes | 1.00 |
| 45 | PersonalInfoUpdate({"username": "John", "password": "123456) | PersonalInfoUpdate(...) | yes | yes | 1.00 |
| 46 | AccountInfo({"username": "Mary", "password": "abcdef) | AccountInfo(...) | yes | yes | 1.00 |
| 46 | PersonalInfoUpdate({"username": "Mary", "password": "abcdef) | PersonalInfoUpdate(...) | yes | yes | 1.00 |
| 47 | AccountInfo({"username": "Peter", "password": "qwert) | AccountInfo(...) | yes | yes | 1.00 |
| 47 | PersonalInfoUpdate({"username": "Peter", "password": "qwert) | PersonalInfoUpdate(...) | yes | yes | 1.00 |
| 48 | AccountInfo({"username": "Tom", "password": "asdfgh") | AccountInfo(...) | yes | yes | 1.00 |
| 48 | PersonalInfoUpdate({"username": "Tom", "password": "asdfgh") | PersonalInfoUpdate(...) | yes | yes | 1.00 |
| 49 | AccountInfo({"username": "Jerry", "password": "zxcvb) | AccountInfo(...) | yes | yes | 1.00 |
| 49 | PersonalInfoUpdate({"username": "Jerry", "password": "zxcvb) | PersonalInfoUpdate(...) | yes | yes | 1.00 |