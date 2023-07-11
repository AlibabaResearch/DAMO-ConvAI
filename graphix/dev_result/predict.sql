select count(*) from singer	concert_singer
select count(*) from singer	concert_singer
select name, country, age from singer order by age desc	concert_singer
select name, country, age from singer order by age desc	concert_singer
select avg(age), min(age), max(age) from singer where country = 'France'	concert_singer
select avg(age), min(age), max(age) from singer where country = "France"	concert_singer
select song_name, song_release_year from singer order by age asc limit 1	concert_singer
select song_name, song_release_year from singer order by age asc limit 1	concert_singer
select distinct country from singer where age > 20	concert_singer
select distinct country from singer where age > 20	concert_singer
select country, count(*) from singer group by country	concert_singer
select country, count(*) from singer group by country	concert_singer
select song_name from singer where age > (select avg(age) from singer)	concert_singer
select song_name from singer where age > (select avg(age) from singer)	concert_singer
select location, name from stadium where capacity between 5000 and 10000	concert_singer
select location, name from stadium where capacity between 5000 and 10000	concert_singer
select max(capacity), avg(capacity) from stadium	concert_singer
select avg(capacity), max(capacity) from stadium	concert_singer
select name, capacity from stadium order by average desc limit 1	concert_singer
select name, capacity from stadium order by average desc limit 1	concert_singer
select count(*) from concert where year = 2014 or year = 2015	concert_singer
select count(*) from concert where year = 2014 or year = 2015	concert_singer
select stadium.name, count(*) from stadium join concert on stadium.stadium_id = concert.stadium_id group by concert.stadium_id	concert_singer
select stadium.name, count(*) from stadium join concert on stadium.stadium_id = concert.stadium_id group by stadium.stadium_id	concert_singer
select stadium.name, stadium.capacity from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year >= 2014 group by concert.stadium_id order by count(*) desc limit 1	concert_singer
select stadium.name, stadium.capacity from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year > 2013 group by concert.stadium_id order by count(*) desc limit 1	concert_singer
select year from concert group by year order by count(*) desc limit 1	concert_singer
select year from concert group by year order by count(*) desc limit 1	concert_singer
select name from stadium where stadium_id not in (select stadium_id from concert)	concert_singer
select name from stadium where stadium_id not in (select stadium_id from concert)	concert_singer
select country from singer where age > 40 intersect select country from singer where age < 30	concert_singer
select name from stadium except select stadium.name from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2014	concert_singer
select name from stadium except select stadium.name from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2014	concert_singer
select concert.concert_name, concert.theme, count(*) from singer_in_concert join concert on singer_in_concert.concert_id = concert.concert_id group by concert.concert_name	concert_singer
select concert.concert_name, concert.theme, count(*) from singer_in_concert join concert on singer_in_concert.concert_id = concert.concert_id group by concert.concert_name	concert_singer
select singer.name, count(*) from singer_in_concert join singer on singer_in_concert.singer_id = singer.singer_id group by singer.name	concert_singer
select singer.name, count(*) from singer join singer_in_concert on singer.singer_id = singer_in_concert.singer_id group by singer.name	concert_singer
select singer.name from singer_in_concert join concert on singer_in_concert.concert_id = concert.concert_id join singer on singer_in_concert.singer_id = singer.singer_id where concert.year = 2014	concert_singer
select singer.name from singer join singer_in_concert on singer.singer_id = singer_in_concert.singer_id join concert on singer_in_concert.concert_id = concert.concert_id where concert.year = 2014	concert_singer
select name, country from singer where song_name like "%hey%"	concert_singer
select name, country from singer where song_name like '%hey%'	concert_singer
select stadium.name, stadium.location from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2014 intersect select stadium.name, stadium.location from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2015	concert_singer
select stadium.name, stadium.location from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2014 intersect select stadium.name, stadium.location from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2015	concert_singer
select count(*) from stadium join concert on stadium.stadium_id = concert.stadium_id where stadium.capacity = (select max(capacity) from stadium)	concert_singer
select count(*) from concert join stadium on concert.stadium_id = stadium.stadium_id where stadium.capacity = (select max(capacity) from stadium)	concert_singer
select count(*) from pets where weight > 10	pets_1
select count(*) from pets where weight > 10	pets_1
select weight from pets where pettype = 'dog' order by pet_age limit 1	pets_1
select weight from pets order by pet_age limit 1	pets_1
select max(weight), pettype from pets group by pettype	pets_1
select pettype, max(weight) from pets group by pettype	pets_1
select count(*) from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where student.age > 20	pets_1
select count(*) from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where student.age > 20	pets_1
select count(*) from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where student.sex = "F" and pets.pettype = "dog"	pets_1
select count(*) from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where student.sex = "F" and pets.pettype = "dog"	pets_1
select count(distinct pettype) from pets	pets_1
select count(distinct pettype) from pets	pets_1
select distinct student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = "cat" or pets.pettype = "dog"	pets_1
select distinct student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = "cat" or pets.pettype = "dog"	pets_1
select student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = "cat" intersect select student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = "dog"	pets_1
select student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = "cat" intersect select student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = "dog"	pets_1
select major, age from student where stuid not in (select has_pet.stuid from has_pet join pets on has_pet.petid = pets.petid where pets.pettype = "cat")	pets_1
select major, age from student where stuid not in (select has_pet.stuid from has_pet join pets on has_pet.petid = pets.petid where pets.pettype = "cat")	pets_1
select stuid from student except select stuid from has_pet join pets on has_pet.petid = pets.petid where pets.pettype = "cat"	pets_1
select stuid from student except select stuid from has_pet join pets on has_pet.petid = pets.petid where pets.pettype = "cat"	pets_1
select fname, age from student where stuid in (select stuid from has_pet join pets on has_pet.petid = pets.petid where pets.pettype = "dog" except select stuid from has_pet join pets on has_pet.petid = pets.petid where pets.pettype = "cat")	pets_1
select student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = "dog" except select student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = "cat"	pets_1
select pettype, weight from pets order by pet_age limit 1	pets_1
select pettype, weight from pets order by pet_age limit 1	pets_1
select petid, weight from pets where pet_age > 1	pets_1
select petid, weight from pets where pet_age > 1	pets_1
select pettype, avg(pet_age), max(pet_age) from pets group by pettype	pets_1
select pettype, avg(pet_age), max(pet_age) from pets group by pettype	pets_1
select pettype, avg(weight) from pets group by pettype	pets_1
select pettype, avg(weight) from pets group by pettype	pets_1
select distinct student.fname, student.age from student join has_pet on student.stuid = has_pet.stuid	pets_1
select distinct student.fname, student.age from student join has_pet on student.stuid = has_pet.stuid	pets_1
select has_pet.petid from has_pet join student on has_pet.stuid = student.stuid where student.lname = "Smith"	pets_1
select has_pet.petid from has_pet join student on has_pet.stuid = student.stuid where student.lname = "Smith"	pets_1
select count(*), stuid from has_pet group by stuid	pets_1
select count(*), student.stuid from student join has_pet on student.stuid = has_pet.stuid group by student.stuid	pets_1
select student.fname, student.sex from student join has_pet on student.stuid = has_pet.stuid group by has_pet.stuid having count(*) > 1	pets_1
select student.fname, student.sex from student join has_pet on student.stuid = has_pet.stuid group by has_pet.stuid having count(*) > 1	pets_1
select student.lname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pet_age = 3	pets_1
select student.lname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pet_age = 3	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet)	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet)	pets_1
select count(*) from continents	car_1
select count(*) from continents	car_1
select continents.continent, countries.countryname, count(*) from continents join countries on continents.continent = countries.continent group by continents.continent	car_1
select continents.continent, countries.countryname, count(*) from continents join countries on continents.continent = countries.continent group by continents.continent	car_1
select count(*) from countries	car_1
select count(*) from countries	car_1
select car_makers.fullname, car_makers.id, count(*) from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id	car_1
select car_makers.fullname, car_makers.id, count(*) from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id	car_1
select car_names.model from cars_data join car_names on cars_data.id = car_names.makeid order by cars_data.horsepower limit 1	car_1
select car_names.model from cars_data join car_names on cars_data.id = car_names.makeid order by cars_data.horsepower limit 1	car_1
select car_names.model from cars_data join car_names on cars_data.id = car_names.makeid where cars_data.weight < (select avg(weight) from cars_data)	car_1
select car_names.model from cars_data join car_names on cars_data.id = car_names.makeid where cars_data.weight < (select avg(weight) from cars_data)	car_1
select distinct car_makers.maker from car_makers join cars_data on car_makers.id = cars_data.id where cars_data.year = 1970	car_1
select distinct car_makers.maker from car_makers join car_names on car_makers.id = car_names.makeid join cars_data on cars_data.id = car_names.makeid where cars_data.year = 1970	car_1
select car_names.make, cars_data.year from cars_data join car_names on cars_data.id = car_names.makeid order by cars_data.year limit 1	car_1
select car_makers.maker, cars_data.year from cars_data join model_list on cars_data.id = model_list.modelid join car_makers on cars_data.id = car_makers.id order by cars_data.year limit 1	car_1
select distinct car_names.model from car_names join cars_data on car_names.makeid = cars_data.id where cars_data.year > 1980	car_1
select distinct model_list.model from model_list join cars_data on model_list.modelid = cars_data.id where cars_data.year > 1980	car_1
select continents.continent, count(*) from continents join car_makers on continents.contid = car_makers.country group by continents.continent	car_1
select countries.countryname, count(*) from countries join car_makers on countries.countryid = car_makers.country group by countries.continent	car_1
select countries.countryname from car_makers join countries on car_makers.country = countries.countryid group by countries.countryname order by count(*) desc limit 1	car_1
select countries.countryname from car_makers join countries on car_makers.country = countries.countryid group by countries.countryname order by count(*) desc limit 1	car_1
select count(*), car_makers.fullname from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id	car_1
select count(*), car_makers.id, car_makers.fullname from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id	car_1
select cars_data.accelerate from car_names join cars_data on car_names.makeid = cars_data.id where car_names.make = "amc hornet" and car_names.make = "sw"	car_1
select cars_data.accelerate from car_names join cars_data on car_names.makeid = cars_data.id where car_names.make = "amc hornet" and car_names.make = "sw"	car_1
select count(*) from car_makers join countries on car_makers.country = countries.countryid where countries.countryname = "france"	car_1
select count(*) from car_makers join countries on car_makers.country = countries.countryid where countries.countryname = "France"	car_1
select count(*) from countries join car_makers on countries.countryid = car_makers.country where countries.countryname = "USA"	car_1
select count(*) from car_makers where country = 'USA'	car_1
select avg(mpg) from cars_data where cylinders = 4	car_1
select avg(mpg) from cars_data where cylinders = 4	car_1
select min(weight) from cars_data where cylinders = 8 and year = 1974	car_1
select min(weight) from cars_data where cylinders = 8 and year = 1974	car_1
select maker, model from model_list	car_1
select maker, model from model_list	car_1
select countries.countryname, car_makers.country from car_makers join countries on car_makers.country = countries.countryid	car_1
select countries.countryname, car_makers.country from countries join car_makers on countries.countryid = car_makers.country	car_1
select count(*) from cars_data where horsepower > 150	car_1
select count(*) from cars_data where horsepower > 150	car_1
select avg(weight), year from cars_data group by year	car_1
select avg(weight), year from cars_data group by year	car_1
select countries.countryname from car_makers join countries on car_makers.country = countries.countryid join continents on continents.continent =	car_1
select countries.countryname from car_makers join countries on car_makers.country = countries.countryid group by countries.countryname having count(*) >= 3	car_1
select max(horsepower), car_names.make from cars_data join car_names on cars_data.id = car_names.model where cars_data.cylinders = 3	car_1
select car_names.make, max(cars_data.horsepower) from cars_data join car_names on cars_data.cylinders = car_names.makeid where cars_data.cylinders = 3	car_1
select model_list.model from cars_data join model_list on cars_data.id = model_list.modelid group by model_list.model order by sum(cars_data.mpg) desc limit 1	car_1
select car_names.model from cars_data join car_names on cars_data.id = car_names.makeid order by cars_data.mpg desc limit 1	car_1
select avg(horsepower) from cars_data where year < 1980	car_1
select avg(horsepower) from cars_data where year < 1980	car_1
select avg(edispl) from cars_data join model_list on cars_	car_1
select avg(edispl) from cars_data	car_1
select max(accelerate), cylinders from cars_data group by cylinders	car_1
select max(accelerate), cylinders from cars_data group by cylinders	car_1
select model from car_names group by model order by count(*) desc limit 1	car_1
select model from model_list group by model order by count(*) desc limit 1	car_1
select count(*) from cars_data where cylinders > 4	car_1
select count(*) from cars_data where cylinders > 4	car_1
select count(*) from cars_data where year = 1980	car_1
select count(*) from cars_data where year = 1980	car_1
select count(*) from car_makers join model_list on car_makers.id = model_list.maker where car_makers.fullname = "American motor Company"	car_1
select count(*) from car_makers join model_list on car_makers.id = model_list.maker where car_makers.fullname = "American motor Company"	car_1
select car_makers.fullname, car_makers.id from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id having count(*) > 3	car_1
select car_makers.fullname, model_list.maker from car_makers join model_list on car_makers.id = model_list.maker group by model_list.maker having count(*) > 3	car_1
select distinct model_list.model from model_list join car_makers on model_list.maker = car_makers.id join cars_data on cars_data.id = car_makers.id where car_makers.fullname = 'General Motors' or cars_data.weight > 3500	car_1
select distinct model_list.model from model_list join car_makers on model_list.maker = car_makers.id join cars_data on cars_data.id = car_makers.id where car_makers.fullname = 'General Motors' or cars_data.weight > 3500	car_1
select year from cars_data where weight <= 3000 and weight <= 4000	car_1
select distinct year from cars_data where weight < 4000 intersect select distinct year from cars_data where weight > 3000	car_1
select horsepower from cars_data order by accelerate desc limit 1	car_1
select horsepower from cars_data order by accelerate desc limit 1	car_1
select cylinders from cars_data where model = 'Volvo' order by accelerate limit 1;.030 sec	car_1
select cars_data.cylinders from cars_data join model_list on cars_data.id = model_list.modelid where model_list.model = "Volvo" order by cars_data.accelerate limit 1	car_1
select count(*) from cars_data where accelerate > (select max(accelerate) from cars_data)	car_1
select count(*) from cars_data where accelerate > (select max(accelerate) from cars_data)	car_1
select count(*) from (select country from car_makers group by country having count(*) > 2)	car_1
select count(*) from (select country from car_makers group by country having count(*) > 2)	car_1
select count(*) from cars_data where cylinders > 6	car_1
select count(*) from cars_data where cylinders > 6	car_1
select car_names.model from cars_data join car_names on cars_data.id = car_names.model where cars_data.cylinders = 4 order by cars_data.horsepower desc limit 1	car_1
select car_names.model from cars_data join car_names on cars_data.id = car_names.model where cars_data.cylinders = 4 order by horsepower desc limit 1	car_1
select car_names.makeid, car_names.make from cars_data join car_names on cars_data.id = car_names.makeid where cars_data.horsepower > (select min(horsepower) from cars_data) and car_names.makeid not in (select id from cars_data where cylinders > 3)	car_1
select car_names.makeid, car_names.make from cars_data join car_names on cars_data.id = car_names.makeid where cars_data.cylinders < 4	car_1
select max(mpg) from cars_data where cylinders = 8 or year < 1980	car_1
select max(mpg) from cars_data where cylinders = 8 or year < 1980	car_1
error
select distinct model_list.model from model_list join cars_data on model_list.modelid = cars_data.id join car_makers on model_list.maker = car_makers.id where car_makers.fullname!= "Ford Motor Company" and cars_data.weight < 3500	car_1
select countryname from countries except select countries.countryname from countries join car_makers on countries.countryid = car_makers.country	car_1
select countryname from countries except select countries.countryname from countries join car_makers on countries.countryid = car_makers.country	car_1
select car_makers.id, car_makers.maker from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id having count(*) >= 2 intersect select car_makers.id, car_makers.maker from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id having count(*) > 3	car_1
select car_makers.id, car_makers.maker from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id having count(*) >= 2 intersect select car_makers.id, car_makers.maker from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id having count(*) > 3	car_1
select countries.countryid, countries.countryname from car_makers join countries on car_makers.country = countries.countryid group by countries.countryid having count(*) > 3 union select countries.countryid, countries.countryname from car_makers join countries on car_makers.country = countries.countryid where car_makers.maker = "Fiat"	car_1
select countries.countryid, countries.countryname from countries join car_makers on countries.countryid = car_makers.country where car_makers.maker = "Fiat" group by countries.countryid having count(*) > 3 union select countries.countryid, countries.countryname from countries join model_list on countries.countryid = model_list.maker where model_list.model = "Fiat"	car_1
select country from airlines where airline = 'JetBlue Airways'	flight_2
select country from airlines where airline = 'JetBlue Airways'	flight_2
select abbreviation from airlines where airline = 'JetBlue Airways'	flight_2
select abbreviation from airlines where airline = 'JetBlue Airways'	flight_2
select airline, abbreviation from airlines where country = 'USA'	flight_2
select airline, abbreviation from airlines where country = 'USA'	flight_2
select airportcode, airportname from airports where city = 'Anthony'	flight_2
select airportcode, airportname from airports where city = 'Anthony'	flight_2
select count(*) from airlines	flight_2
select count(*) from airlines	flight_2
select count(*) from airports	flight_2
select count(*) from airports	flight_2
select count(*) from flights	flight_2
select count(*) from flights	flight_2
select airline from airlines where abbreviation = 'UAL'	flight_2
select airline from airlines where abbreviation = 'UAL'	flight_2
select count(*) from airlines where country = 'USA'	flight_2
select count(*) from airlines where country = 'USA'	flight_2
select city, country from airports where airportname = 'Alton'	flight_2
select city, country from airports where airportname = 'Alton'	flight_2
select airportname from airports where airportcode = 'AKO'	flight_2
select airportname from airports where airportcode = 'AKO'	flight_2
select airportname from airports where city = 'Aberdeen'	flight_2
select airportname from airports where city = 'Aberdeen'	flight_2
select count(*) from flights where destairport = 'APG'	flight_2
select count(*) from flights where destairport = "APG"	flight_2
select count(*) from airports join flights on airports.airportcode = flights.destairport where airports.airportcode = 'ATO'	flight_2
select count(*) from airports join flights on airports.airportcode = flights.sourceairport where airports.airportcode = 'ATO'	flight_2
select count(*) from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen'	flight_2
select count(*) from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen'	flight_2
select count(*) from airports join flights on airports.airportcode = flights.destairport where airports.city = 'Aberdeen'	flight_2
select count(*) from airports join flights on airports.airportcode = flights.destairport where airports.city = 'Aberdeen'	flight_2
select count(*) from flights where sourceairport = 'Aberdeen' and destairport = 'Ashley'	flight_2
select count(*) from flights join airports on flights.sourceairport = airports.airportcode where airports.city = 'Aberdeen' and airports.airportname = 'Ashley'	flight_2
select count(*) from airlines join flights on airlines.uid = flights.airline where airlines.airline = "JetBlue Airways"	flight_2
select count(*) from airlines join flights on airlines.uid = flights.airline where airlines.airline = 'JetBlue Airways'	flight_2
select count(*) from flights join airports on flights.sourceairport = airports.airportcode join airlines on flights.airline = airlines.uid where airports.airportcode = 'ASY' and airlines.airline = 'United Airlines'	flight_2
select count(*) from flights join airports on flights.destairport = airports.airportcode join airlines on flights.airline = airlines.uid where airports.airportcode = "ASY" and airlines.airline = "United Airlines"	flight_2
select count(*) from flights join airports on flights.sourceairport = airports.airportcode join airlines on flights.airline = airlines.uid where airports.airportcode = 'AHD' and airlines.airline = 'United Airlines'	flight_2
select count(*) from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = 'AHD' and airlines.airline = 'United Airlines'	flight_2
select count(*) from airports join flights on airports.airportcode = flights.sourceairport join airlines on flights.airline = airlines.uid where airports.city = 'Aberdeen' and airlines.airline = 'United Airlines'	flight_2
select count(*) from airports join flights on airports.airportcode = flights.destairport join airlines on flights.airline = airlines.uid where airports.city = 'Aberdeen' and airlines.airline = 'United Airlines'	flight_2
select airports.city from airports join flights on airports.airportcode = flights.sourceairport group by airports.city order by count(*) desc limit 1	flight_2
select airports.city from airports join flights on airports.airportcode = flights.destairport group by airports.city order by count(*) desc limit 1	flight_2
select airports.city from airports join flights on airports.airportcode = flights.destairport group by airports.city order by count(*) desc limit 1	flight_2
select airports.city from airports join flights on airports.airportcode = flights.sourceairport group by flights.sourceairport order by count(*) desc limit 1	flight_2
select airports.airportcode from airports join flights on airports.airportcode = flights.sourceairport group by flights.sourceairport order by count(*) desc limit 1	flight_2
select airports.airportcode from airports join flights on airports.airportcode = flights.sourceairport group by flights.sourceairport order by count(*) desc limit 1	flight_2
select airportcode from airports group by airportcode order by count(*) limit 1	flight_2
select airportcode from airports group by airportcode order by count(*) limit 1	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline group by airlines.airline order by count(*) desc limit 1	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline group by airlines.airline order by count(*) desc limit 1	flight_2
select airlines.abbreviation, airlines.country from airlines join flights on airlines.uid = flights.airline group by flights.airline order by count(*) limit 1	flight_2
select airlines.abbreviation, airlines.country from airlines join flights on airlines.uid = flights.airline group by airlines.uid order by count(*) limit 1	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.destairport = 'AHD'	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = 'AHD'	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.destairport = 'AHD'	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.destairport = 'AHD'	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = 'APG' intersect select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = 'CVO'	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.destairport = 'APG' intersect select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = 'CVO'	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = "CVO" except select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = "APG"	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = 'CVO' except select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.destairport = 'APG'	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline group by airlines.airline having count(*) >= 10	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline group by airlines.airline having count(*) >= 10	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline group by airlines.uid having count(*) < 200	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airline group by airlines.airline having count(*) < 200	flight_2
select flights.flightno from airlines join flights on airlines.uid = flights.airline where airlines.airline = "United Airlines"	flight_2
select flights.flightno from airlines join flights on airlines.uid = flights.airline where airlines.airline = 'United Airlines'	flight_2
select flightno from flights where destairport = 'APG'	flight_2
select flightno from flights where sourceairport = 'APG'	flight_2
select flightno from flights where destairport = 'APG'	flight_2
select flightno from flights where destairport = 'APG'	flight_2
select flights.flightno from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen'	flight_2
select flights.flightno from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen'	flight_2
select flights.flightno from airports join flights on airports.airportcode = flights.destairport where airports.city = 'Aberdeen'	flight_2
select flights.flightno from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen'	flight_2
select count(*) from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen' or airports.city = 'Abilene'	flight_2
select count(*) from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen' or airports.city = 'Abilene'	flight_2
select airportname from airports where airportcode not in (select sourceairport from flights)	flight_2
select airportname from airports except select airports.airportname from airports join flights on airports.airportcode = flights.sourceairport or flights.destairport = flights.destairport	flight_2
select count(*) from employee	employee_hire_evaluation
select count(*) from employee	employee_hire_evaluation
select name from employee order by age asc	employee_hire_evaluation
select name from employee order by age asc	employee_hire_evaluation
select city, count(*) from employee group by city	employee_hire_evaluation
select count(*), city from employee group by city	employee_hire_evaluation
select city from employee where age < 30 group by city having count(*) > 1	employee_hire_evaluation
select city from employee where age < 30 group by city having count(*) > 1	employee_hire_evaluation
select location, count(*) from shop group by location	employee_hire_evaluation
select location, count(*) from shop group by location	employee_hire_evaluation
select manager_name, district from shop order by number_products desc limit 1	employee_hire_evaluation
select manager_name, district from shop order by number_products desc limit 1	employee_hire_evaluation
select min(number_products), max(number_products) from shop	employee_hire_evaluation
select min(number_products), max(number_products) from shop	employee_hire_evaluation
select name, location, district from shop order by number_products desc	employee_hire_evaluation
select name, location, district from shop order by number_products desc	employee_hire_evaluation
select name from shop where number_products > (select avg(number_products) from shop)	employee_hire_evaluation
select name from shop where number_products > (select avg(number_products) from shop)	employee_hire_evaluation
select employee.name from employee join evaluation on employee.employee_id = evaluation.employee_id group by employee.employee_id order by count(*) desc limit 1	employee_hire_evaluation
select employee.name from employee join evaluation on employee.employee_id = evaluation.employee_id group by employee.employee_id order by sum(evaluation.year_awarded) desc limit 1	employee_hire_evaluation
select employee.name from employee join evaluation on employee.employee_id = evaluation.employee_id order by evaluation.bonus desc limit 1	employee_hire_evaluation
select employee.name from employee join evaluation on employee.employee_id = evaluation.employee_id order by evaluation.bonus desc limit 1	employee_hire_evaluation
select name from employee where employee_id not in (select employee_id from evaluation)	employee_hire_evaluation
select name from employee where employee_id not in (select employee_id from evaluation)	employee_hire_evaluation
select shop.name from shop join hiring on shop.shop_id = hiring.shop_id group by hiring.shop_id order by count(*) desc limit 1	employee_hire_evaluation
select shop.name from shop join hiring on shop.shop_id = hiring.shop_id group by hiring.shop_id order by count(*) desc limit 1	employee_hire_evaluation
select name from shop where shop_id not in (select shop_id from hiring)	employee_hire_evaluation
select name from shop where shop_id not in (select shop_id from hiring)	employee_hire_evaluation
select shop.name, count(*) from hiring join shop on hiring.shop_id = shop.shop_id group by shop.name	employee_hire_evaluation
select shop.name, count(*) from hiring join shop on hiring.shop_id = shop.shop_id group by shop.name	employee_hire_evaluation
select sum(bonus) from evaluation	employee_hire_evaluation
select sum(bonus) from evaluation	employee_hire_evaluation
select * from hiring	employee_hire_evaluation
select * from hiring	employee_hire_evaluation
select district from shop where number_products < 3000 intersect select district from shop where number_products > 10000	employee_hire_evaluation
select district from shop where number_products < 3000 intersect select district from shop where number_products > 10000	employee_hire_evaluation
select count(distinct location) from shop	employee_hire_evaluation
select count(distinct location) from shop	employee_hire_evaluation
select count(*) from documents	cre_Doc_Template_Mgt
select count(*) from documents	cre_Doc_Template_Mgt
select document_id, document_name, document_description from documents	cre_Doc_Template_Mgt
select document_id, document_name, document_description from documents	cre_Doc_Template_Mgt
select document_name, template_id from documents where document_description like '%w%'	cre_Doc_Template_Mgt
select document_name, template_id from documents where document_description like '%w%'	cre_Doc_Template_Mgt
select document_id, template_id, document_description from documents where document_name = "Robbin CV"	cre_Doc_Template_Mgt
select document_id, template_id, document_description from documents where document_name = "Robbin CV"	cre_Doc_Template_Mgt
select count(distinct template_id) from documents	cre_Doc_Template_Mgt
select count(distinct template_id) from documents	cre_Doc_Template_Mgt
select count(*) from documents join templates on documents.template_id = templates.template_id where templates.template_type_code = "PPT"	cre_Doc_Template_Mgt
select count(*) from documents join templates on documents.template_id = templates.template_id where templates.template_type_code = "PPT"	cre_Doc_Template_Mgt
select template_id, count(*) from documents group by template_id	cre_Doc_Template_Mgt
select template_id, count(*) from documents group by template_id	cre_Doc_Template_Mgt
select documents.template_id, templates.template_type_code from templates join documents on templates.template_id = documents.template_id group by documents.template_id order by count(*) desc limit 1	cre_Doc_Template_Mgt
select documents.template_id, templates.template_type_code from templates join documents on templates.template_id = documents.template_id group by documents.template_id order by count(*) desc limit 1	cre_Doc_Template_Mgt
select template_id from documents group by template_id having count(*) > 1	cre_Doc_Template_Mgt
select template_id from documents group by template_id having count(*) > 1	cre_Doc_Template_Mgt
select template_id from templates except select template_id from documents	cre_Doc_Template_Mgt
select template_id from templates except select template_id from documents	cre_Doc_Template_Mgt
select count(*) from templates	cre_Doc_Template_Mgt
select count(*) from templates	cre_Doc_Template_Mgt
select template_id, version_number, template_type_code from templates	cre_Doc_Template_Mgt
select template_id, version_number, template_type_code from templates	cre_Doc_Template_Mgt
select distinct template_type_code from templates	cre_Doc_Template_Mgt
select distinct template_type_code from templates	cre_Doc_Template_Mgt
select template_id from templates where template_type_code = "PP" or template_type_code = "PPT"	cre_Doc_Template_Mgt
select template_id from templates where template_type_code = "PP" or template_type_code = "PPT"	cre_Doc_Template_Mgt
select count(*) from templates where template_type_code = "CV"	cre_Doc_Template_Mgt
select count(*) from templates where template_type_code = "CV"	cre_Doc_Template_Mgt
select version_number, template_type_code from templates where version_number > 5	cre_Doc_Template_Mgt
select version_number, template_type_code from templates where version_number > 5	cre_Doc_Template_Mgt
select template_type_code, count(*) from templates group by template_type_code	cre_Doc_Template_Mgt
select template_type_code, count(*) from templates group by template_type_code	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code order by count(*) desc limit 1	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code order by count(*) desc limit 1	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code having count(*) < 3	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code having count(*) < 3	cre_Doc_Template_Mgt
select version_number, template_type_code from templates order by version_number asc limit 1	cre_Doc_Template_Mgt
select version_number, template_type_code from templates order by version_number asc limit 1	cre_Doc_Template_Mgt
select templates.template_type_code from documents join templates on documents.template_id = templates.template_id where documents.document_name = "Data base"	cre_Doc_Template_Mgt
select templates.template_type_code from documents join templates on documents.template_id = templates.template_id where documents.document_name = "Data base"	cre_Doc_Template_Mgt
select documents.document_name from documents join templates on documents.template_id = templates.template_id where templates.template_type_code = "BK"	cre_Doc_Template_Mgt
select documents.document_name from documents join templates on documents.template_id = templates.template_id where templates.template_type_code = "BK"	cre_Doc_Template_Mgt
select templates.template_type_code, count(*) from templates join documents on templates.template_id = documents.template_id group by templates.template_type_code	cre_Doc_Template_Mgt
select templates.template_type_code, count(*) from templates join documents on templates.template_id = documents.template_id group by templates.template_type_code	cre_Doc_Template_Mgt
select templates.template_type_code from templates join documents on templates.template_id = documents.template_id group by templates.template_type_code order by count(*) desc limit 1	cre_Doc_Template_Mgt
select templates.template_type_code from templates join documents on templates.template_id = documents.template_id group by templates.template_type_code order by count(*) desc limit 1	cre_Doc_Template_Mgt
select template_type_code from ref_template_types except select template_type_code from templates	cre_Doc_Template_Mgt
select template_type_code from templates except select templates.template_type_code from templates join documents on templates.template_id = documents.template_id	cre_Doc_Template_Mgt
select template_type_code, template_type_description from ref_template_types	cre_Doc_Template_Mgt
select template_type_code, template_type_description from ref_template_types	cre_Doc_Template_Mgt
select template_type_description from ref_template_types where template_type_code = "AD"	cre_Doc_Template_Mgt
select template_type_description from ref_template_types where template_type_code = "AD"	cre_Doc_Template_Mgt
select template_type_code from ref_template_types where template_type_description = "Book"	cre_Doc_Template_Mgt
select template_type_code from ref_template_types where template_type_description = "Book"	cre_Doc_Template_Mgt
select distinct ref_template_types.template_type_description from ref_template_types join templates on ref_template_types.template_type_code = templates.template_type_code join documents on documents.template_id = templates.template_id	cre_Doc_Template_Mgt
select distinct ref_template_types.template_type_description from ref_template_types join documents on ref_template_types.template_type_code = documents.template_id	cre_Doc_Template_Mgt
select templates.template_id from templates join ref_template_types on templates.template_type_code = ref_template_types.template_type_code where ref_template_types.template_type_description = "Presentation"	cre_Doc_Template_Mgt
select templates.template_id from templates join ref_template_types on templates.template_type_code = ref_template_types.template_type_code where ref_template_types.template_type_description = "Presentation"	cre_Doc_Template_Mgt
select count(*) from paragraphs	cre_Doc_Template_Mgt
select count(*) from paragraphs	cre_Doc_Template_Mgt
select count(*) from documents join paragraphs on documents.document_id = paragraphs.document_id where documents.document_name = "Summer Show"	cre_Doc_Template_Mgt
select count(*) from documents join paragraphs on documents.document_id = paragraphs.document_id where documents.document_name = "Summer Show"	cre_Doc_Template_Mgt
select other_details from paragraphs where paragraph_text = "Korea"	cre_Doc_Template_Mgt
select other_details from paragraphs where paragraph_text like "%korea%"	cre_Doc_Template_Mgt
select paragraphs.paragraph_id, paragraphs.paragraph_text from documents join paragraphs on documents.document_id = paragraphs.document_id where documents.document_name = "Welcome to NY"	cre_Doc_Template_Mgt
select paragraphs.paragraph_id, paragraphs.paragraph_text from paragraphs join documents on paragraphs.document_id = documents.document_id where documents.document_name = "Welcome to NY"	cre_Doc_Template_Mgt
select paragraphs.paragraph_text from documents join paragraphs on documents.document_id = paragraphs.document_id where documents.document_name = "Customer reviews"	cre_Doc_Template_Mgt
select paragraphs.paragraph_text from documents join paragraphs on documents.document_id = paragraphs.document_id where documents.document_name = "Customer reviews"	cre_Doc_Template_Mgt
select document_id, count(*) from paragraphs group by document_id order by document_id	cre_Doc_Template_Mgt
select document_id, count(*) from paragraphs group by document_id order by document_id	cre_Doc_Template_Mgt
select documents.document_id, documents.document_name, count(*) from paragraphs join documents on paragraphs.document_id = documents.document_id group by documents.document_id	cre_Doc_Template_Mgt
select documents.document_id, documents.document_name, count(*) from paragraphs join documents on paragraphs.document_id = documents.document_id group by documents.document_id	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id having count(*) >= 2	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id having count(*) >= 2	cre_Doc_Template_Mgt
select documents.document_id, documents.document_name from documents join paragraphs on documents.document_id = paragraphs.document_id group by documents.document_id order by count(*) desc limit 1	cre_Doc_Template_Mgt
select documents.document_id, documents.document_name from documents join paragraphs on documents.document_id = paragraphs.document_id group by documents.document_id order by count(*) desc limit 1	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id order by count(*) asc limit 1	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id order by count(*) asc limit 1	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id having count(*) between 1 and 2	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id having count(*) between 1 and 2	cre_Doc_Template_Mgt
select document_id from paragraphs where paragraph_text = "Brazil" intersect select document_id from paragraphs where paragraph_text = "Ireland"	cre_Doc_Template_Mgt
select document_id from paragraphs where paragraph_text = "Brazil" intersect select document_id from paragraphs where paragraph_text = "Ireland"	cre_Doc_Template_Mgt
select count(*) from teacher	course_teach
select count(*) from teacher	course_teach
select name from teacher order by age asc	course_teach
select name from teacher order by age asc	course_teach
select age, hometown from teacher	course_teach
select age, hometown from teacher	course_teach
select name from teacher where hometown!= "Little lever Urban District"	course_teach
select name from teacher where hometown!= "Little lever Urban District"	course_teach
select name from teacher where age = 32 or age = 33	course_teach
select name from teacher where age = 32 or age = 33	course_teach
select hometown from teacher order by age asc limit 1	course_teach
select hometown from teacher order by age asc limit 1	course_teach
select hometown, count(*) from teacher group by hometown	course_teach
select hometown, count(*) from teacher group by hometown	course_teach
select hometown from teacher group by hometown order by count(*) desc limit 1	course_teach
select hometown from teacher group by hometown order by count(*) desc limit 1	course_teach
select hometown from teacher group by hometown having count(*) >= 2	course_teach
select hometown from teacher group by hometown having count(*) >= 2	course_teach
select teacher.name, course.course_id from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id	course_teach
select teacher.name, course.course from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id	course_teach
select teacher.name, course.course from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id order by teacher.name asc	course_teach
select teacher.name, course.course from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id order by teacher.name	course_teach
select teacher.name from course join course_arrange on course.course_id = course_arrange.course_id join teacher on course_arrange.teacher_id = teacher.teacher_id where course.course = 'Math'	course_teach
select teacher.name from course join course_arrange on course.course_id = course_arrange.course_id join teacher on course_arrange.teacher_id = teacher.teacher_id where course.course = 'Math'	course_teach
select teacher.name, count(*) from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id group by teacher.name	course_teach
select teacher.name, count(*) from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id group by teacher.name	course_teach
select teacher.name from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id group by course_arrange.teacher_id having count(*) >= 2	course_teach
select teacher.name from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id group by course_arrange.teacher_id having count(*) >= 2	course_teach
select name from teacher where teacher_id not in (select teacher_id from course_arrange)	course_teach
select name from teacher where teacher_id not in (select teacher_id from course_arrange)	course_teach
select count(*) from visitor where age < 30	museum_visit
select name from visitor where level_of_membership > 4 order by level_of_membership	museum_visit
select avg(age) from visitor where level_of_membership!= 4	museum_visit
select name, level_of_membership from visitor where level_of_membership > 4 order by age desc	museum_visit
select museum_id, name from museum order by num_of_staff desc limit 1	museum_visit
select avg(num_of_staff) from museum where open_year < 2009	museum_visit
select open_year, num_of_staff from museum where name = 'Plaza Museum'	museum_visit
select name from museum where num_of_staff > (select min(num_of_staff) from museum where open_year > 2010)	museum_visit
select visitor.id, visitor.name, visitor.age from visitor join visit on visitor.id = visit.visitor_id group by visit.visitor_id having count(*) > 1	museum_visit
select visitor.id, visitor.name, visitor.level_of_membership from visitor join visit on visitor.id = visit.visitor_id group by visit.visitor_id order by sum(total_spent) desc limit 1	museum_visit
select museum.museum_id, museum.name from museum join visit on museum.museum_id = visit.museum_id group by visit.museum_id order by count(*) desc limit 1	museum_visit
select name from museum where museum_id not in (select museum_id from visit)	museum_visit
select visitor.name, visitor.age from visitor join visit on visitor.id = visit.visitor_id group by visit.visitor_id order by sum(visit.num_of_ticket) desc limit 1	museum_visit
select avg(num_of_ticket), max(num_of_ticket) from visit	museum_visit
select sum(visit.total_spent) from visit join visitor on visit.visitor_id = visitor.id where visitor.level_of_membership = 1	museum_visit
select visitor.name from visit join museum on visit.museum_id = museum.museum_id join visitor on visit.visitor_id = visitor.id where museum.open_year < 2009 intersect select visitor.name from visit join museum on visit.museum_id = museum.museum_id join visitor on visit.visitor_id = visitor.id where museum.open_year > 2011	museum_visit
select count(*) from visitor where id not in (select visit.visitor_id from visit join museum on visit.museum_id = museum.museum_id where museum.open_year > 2010)	museum_visit
select count(*) from museum where open_year > 2013 or open_year < 2008	museum_visit
select count(*) from players	wta_1
select count(*) from players	wta_1
select count(*) from matches	wta_1
select count(*) from matches	wta_1
select first_name, birth_date from players where country_code = 'USA'	wta_1
select first_name, birth_date from players where country_code = 'USA'	wta_1
select avg(loser_age), avg(winner_age) from matches	wta_1
select avg(loser_age), avg(winner_age) from matches	wta_1
select avg(winner_rank) from matches	wta_1
select avg(winner_rank) from matches	wta_1
select min(loser_rank) from matches	wta_1
select min(loser_rank) from matches	wta_1
select count(distinct country_code) from players	wta_1
select count(distinct country_code) from players	wta_1
select count(distinct loser_name) from matches	wta_1
select count(distinct loser_name) from matches	wta_1
select tourney_name from matches group by tourney_name having count(*) > 10	wta_1
select tourney_name from matches group by tourney_name having count(*) > 10	wta_1
select winner_name from matches where year = 2013 intersect select winner_name from matches where year = 2016	wta_1
select players.first_name, players.last_name from players join matches on players.player_id = matches.winner_id where matches.year = 2013 intersect select players.first_name, players.last_name from players join matches on players.player_id = matches.winner_id where matches.year = 2016	wta_1
select count(*) from matches where year = 2013 or year = 2016	wta_1
select count(*) from matches where year = 2013 or year = 2016	wta_1
select players.country_code, players.first_name from players join matches on players.player_id = matches.winner_id where matches.tourney_name = "WTA Championships" intersect select players.country_code, players.first_name from players join matches on players.player_id = matches.winner_id where matches.tourney_name = "Australian Open"	wta_1
select players.first_name, players.country_code from players join matches on players.player_id = matches.winner_id where matches.tourney_name = "WTA Championships" intersect select players.first_name, players.country_code from players join matches on players.player_id = matches.winner_id where matches.tourney_name = "Australian Open"	wta_1
select first_name, country_code from players order by birth_date desc limit 1	wta_1
select first_name, country_code from players order by birth_date desc limit 1	wta_1
select first_name, last_name from players order by birth_date	wta_1
select first_name, last_name from players order by birth_date	wta_1
select first_name, last_name from players where hand = "left" or hand = "right" order by birth_date	wta_1
select first_name, last_name from players where hand = "left" order by birth_date	wta_1
select players.first_name, players.country_code from players join rankings on players.player_id = rankings.player_id group by rankings.player_id order by count(*) desc limit 1	wta_1
select players.first_name, players.country_code from players join rankings on players.player_id = rankings.player_id group by rankings.player_id order by count(*) desc limit 1	wta_1
select year from matches group by year order by count(*) desc limit 1	wta_1
select year from matches group by year order by count(*) desc limit 1	wta_1
select matches.winner_name, matches.winner_rank_points from matches join rankings on matches.winner_id = rankings.player_id group by matches.winner_id order by count(*) desc limit 1	wta_1
select winner_name, winner_rank_points from matches group by winner_name order by count(*) desc limit 1	wta_1
select matches.winner_name from matches join rankings on matches.winner_id = rankings.player_id where matches.tourney_name = "Australian Open" order by rankings.ranking_points desc limit 1	wta_1
select matches.winner_name from matches join rankings on matches.winner_id = rankings.player_id where matches.tourney_name = "Australian Open" order by rankings.ranking_points desc limit 1	wta_1
select loser_name, winner_name from matches order by minutes desc limit 1	wta_1
select winner_name, loser_name from matches order by minutes desc limit 1	wta_1
select avg(ranking), players.first_name from players join rankings on players.player_id = rankings.player_id group by players.player_id	wta_1
select players.first_name, avg(rankings.ranking) from rankings join players on rankings.player_id = players.player_id group by players.first_name	wta_1
select sum(ranking_points), players.first_name from players join rankings on players.player_id = rankings.player_id group by players.player_id	wta_1
select players.first_name, sum(rankings.ranking_points) from rankings join players on rankings.player_id = players.player_id group by players.first_name	wta_1
select country_code, count(*) from players group by country_code	wta_1
select country_code, count(*) from players group by country_code	wta_1
select country_code from players group by country_code order by count(*) desc limit 1	wta_1
select country_code from players group by country_code order by count(*) desc limit 1	wta_1
select country_code from players group by country_code having count(*) > 50	wta_1
select country_code from players group by country_code having count(*) > 50	wta_1
select ranking_date, sum(tours) from rankings group by ranking_date	wta_1
select ranking_date, sum(tours) from rankings group by ranking_date	wta_1
select count(*), year from matches group by year	wta_1
select count(*), year from matches group by year	wta_1
select winner_name, winner_rank from matches order by winner_age asc limit 3	wta_1
select winner_name, winner_rank from matches order by winner_age asc limit 3	wta_1
select count(*) from players join matches on players.player_id = matches.winner_id where matches.tourney_name = "WTA Championships" and players.hand = "left"	wta_1
select count(*) from players join matches on players.player_id = matches.winner_id where players.hand = "left" and matches.tourney_name = "WTA Championships"	wta_1
select players.first_name, players.country_code, players.birth_date from players join matches on players.player_id = matches.winner_id order by matches.winner_rank_points desc limit 1	wta_1
select players.first_name, players.country_code, players.birth_date from players join matches on players.player_id = matches.winner_id group by matches.winner_id order by sum(matches.winner_rank_points) desc limit 1	wta_1
select count(*), hand from players group by hand	wta_1
select count(*), hand from players group by hand	wta_1
select count(*) from ship where disposition_of_ship = 'Captured'	battle_death
select name, tonnage from ship order by name desc	battle_death
select name, date, result from battle	battle_death
select max(killed), min(killed) from death group by caused_by_ship_id	battle_death
select avg(injured), caused_by_ship_id from death group by caused_by_ship_id	battle_death
select death.killed, death.injured from death join ship on death.caused_by_ship_id = ship.id where ship.tonnage = "t"	battle_death
select name, result from battle where bulgarian_commander!= "Boril"	battle_death
select distinct battle.id, battle.name from battle join ship on battle.id = ship.lost_in_battle where ship.ship_type = "Brig"	battle_death
select battle.id, battle.name from battle join death on battle.id = death.caused_by_ship_id group by battle.id having sum(death.killed) > 10	battle_death
select ship.id, ship.name from ship join death on ship.id = death.caused_by_ship_id group by death.caused_by_ship_id order by sum(injured) desc limit 1	battle_death
select distinct name from battle where bulgarian_commander = "Kaloyan" and latin_commander = "Baldwin I"	battle_death
select count(distinct result) from battle	battle_death
select count(*) from battle where id not in ( select lost_in_battle from ship where tonnage = 225 )	battle_death
select battle.name, battle.date from ship join battle on ship.id = battle.id where ship.name = 'Lettice' and ship.name = 'HMS Atalanta'	battle_death
select name, result, bulgarian_commander from battle except select battle.name, battle.result, battle.bulgarian_commander from battle join ship on battle.id = ship.lost_in_battle where ship.location = 'English Channel'	battle_death
select note from death where note like "%east%"	battle_death
select line_1, line_2 from addresses	student_transcripts_tracking
select line_1, line_2 from addresses	student_transcripts_tracking
select count(*) from courses	student_transcripts_tracking
select count(*) from courses	student_transcripts_tracking
select course_description from courses where course_name = "math"	student_transcripts_tracking
select course_description from courses where course_name = "math"	student_transcripts_tracking
select zip_postcode from addresses where city = "Port Chelsea"	student_transcripts_tracking
select zip_postcode from addresses where city = "Port Chelsea"	student_transcripts_tracking
select departments.department_name, degree_programs.department_id from degree_programs join departments on degree_programs.department_id = departments.department_id group by degree_programs.department_id order by count(*) desc limit 1	student_transcripts_tracking
select departments.department_name, degree_programs.department_id from degree_programs join departments on degree_programs.department_id = departments.department_id group by degree_programs.department_id order by count(*) desc limit 1	student_transcripts_tracking
select count(distinct department_id) from degree_programs	student_transcripts_tracking
select count(distinct department_id) from degree_programs	student_transcripts_tracking
select count(distinct degree_summary_name) from degree_programs	student_transcripts_tracking
select count(distinct degree_program_id) from degree_programs	student_transcripts_tracking
select count(*) from departments join degree_programs on departments.department_id = degree_programs.department_id where departments.department_name = "Engineering"	student_transcripts_tracking
select count(*) from departments join degree_programs on departments.department_id = degree_programs.department_id where departments.department_name = "Engineering"	student_transcripts_tracking
select section_name, section_description from sections	student_transcripts_tracking
select section_name, section_description from sections	student_transcripts_tracking
select courses.course_name, sections.course_id from courses join sections on courses.course_id = sections.course_id group by sections.course_id having count(*) <= 2	student_transcripts_tracking
select courses.course_name, sections.course_id from courses join sections on courses.course_id = sections.course_id group by sections.course_id having count(*) < 2	student_transcripts_tracking
select section_name from sections order by section_name desc	student_transcripts_tracking
select section_name from sections order by section_name desc	student_transcripts_tracking
select semesters.semester_name, student_enrolment.semester_id from student_enrolment join semesters on student_enrolment.semester_id = semesters.semester_id group by student_enrolment.semester_id order by count(*) desc limit 1	student_transcripts_tracking
select semesters.semester_name, student_enrolment.semester_id from student_enrolment join semesters on student_enrolment.semester_id = semesters.semester_id group by student_enrolment.semester_id order by count(*) desc limit 1	student_transcripts_tracking
select department_description from departments where department_name like "%computer%"	student_transcripts_tracking
select department_description from departments where department_name like "%computer%"	student_transcripts_tracking
select students.first_name, students.middle_name, students.last_name, student_enrolment.student_enrolment_id from student_enrolment join students on student_enrolment.student_enrolment_id = students.student_id group by student_enrolment.student_enrolment_id having count(*) = 2	student_transcripts_tracking
select students.first_name, students.middle_name, students.last_name, student_enrolment.student_id from student_enrolment join students on student_enrolment.student_id = students.student_id group by student_enrolment.student_id having count(*) = 2	student_transcripts_tracking
select students.first_name, students.middle_name, students.last_name from degree_programs join student_enrolment on degree_programs.degree_program_id = student_enrolment.degree_program_id join students on student_enrolment.student_enrolment_id = students.student_id where degree_programs.degree_summary_name = "Bachelor"	student_transcripts_tracking
select students.first_name, students.middle_name, students.last_name from student_enrolment join students on student_enrolment.student_enrolment_id = students.student_id where student_enrolment.degree_program_id = "Bachelors"	student_transcripts_tracking
select degree_program_id from student_enrolment group by degree_program_id order by count(*) desc limit 1	student_transcripts_tracking
select degree_programs.degree_summary_name from student_enrolment join degree_programs on student_enrolment.degree_program_id = degree_programs.degree_program_id group by degree_programs.degree_summary_name order by count(*) desc limit 1	student_transcripts_tracking
select degree_programs.degree_program_id, degree_programs.degree_summary_description from degree_programs join student_enrolment on degree_programs.degree_program_id = student_enrolment.degree_program_id group by degree_programs.degree_program_id order by count(*) desc limit 1	student_transcripts_tracking
select degree_programs.degree_program_id, degree_programs.degree_summary_name from degree_programs join student_enrolment on degree_programs.degree_program_id = student_enrolment.degree_program_id group by degree_programs.degree_program_id order by count(*) desc limit 1	student_transcripts_tracking
select students.student_id, students.first_name, students.middle_name, students.last_name, count(*) from students join student_enrolment on students.student_id = student_enrolment.student_id group by students.student_id order by count(*) desc limit 1	student_transcripts_tracking
select students.first_name, students.middle_name, students.last_name, count(*) from students join student_enrolment on students.student_id = student_enrolment.student_id group by students.student_id order by count(*) desc limit 1	student_transcripts_tracking
select semester_name from semesters where semester_id not in (select semester_id from student_enrolment)	student_transcripts_tracking
select semester_name from semesters where semester_id not in (select semester_id from student_enrolment)	student_transcripts_tracking
select course_name from courses join student_enrolment_courses on courses.course_id = student_enrolment_courses.course_id	student_transcripts_tracking
select courses.course_name from courses join student_enrolment_courses on courses.course_id = student_enrolment_courses.course_id	student_transcripts_tracking
select courses.course_name from courses join student_enrolment_courses on courses.course_id = student_enrolment_courses.course_id group by courses.course_name order by count(*) desc limit 1	student_transcripts_tracking
select courses.course_name from courses join student_enrolment_courses on courses.course_id = student_enrolment_courses.course_id group by courses.course_name order by count(*) desc limit 1	student_transcripts_tracking
select last_name from students where current_address_id in (select student_enrolment.student_id from student_enrolment join addresses on student_enrolment.student_enrolment_id = addresses.address_id where addresses.state_province_county = "North Carolina") except select students.last_name from students join student_enrolment on students.student_id = student_enrolment.student_id join addresses on addresses.address_id = student_enrolment.student_enrolment_id where addresses.state_province_county = "North Carolina"	student_transcripts_tracking
select last_name from students where permanent_address_id in (select student_enrolment.student_id from addresses join student_enrolment on addresses.address_id = student_enrolment.student_enrolment_id where addresses.state_province_county = "North Carolina")	student_transcripts_tracking
select transcripts.transcript_date, transcripts.transcript_id from transcripts join transcript_contents on transcripts.transcript_id = transcript_contents.transcript_id group by transcripts.transcript_id having count(*) >= 2	student_transcripts_tracking
select transcripts.transcript_date, transcripts.transcript_id from transcripts join transcript_contents on transcripts.transcript_id = transcript_contents.transcript_id group by transcripts.transcript_id having count(*) >= 2	student_transcripts_tracking
select cell_mobile_number from students where first_name = "Timmothy" and last_name = "Ward"	student_transcripts_tracking
select cell_mobile_number from students where first_name = "Timmothy" and last_name = "Ward"	student_transcripts_tracking
select students.first_name, students.middle_name, students.last_name from students join student_enrolment on students.student_id = student_enrolment.student_id order by students.date_first_registered asc limit 1	student_transcripts_tracking
select students.first_name, students.middle_name, students.last_name from student_enrolment join students on student_enrolment.student_id = students.student_id order by students.date_first_registered limit 1	student_transcripts_tracking
select first_name, middle_name, last_name from students order by date_first_registered limit 1	student_transcripts_tracking
select first_name, middle_name, last_name from students order by date_first_registered limit 1	student_transcripts_tracking
select distinct first_name from students where permanent_address_id!= (select current_address_id from students)	student_transcripts_tracking
select first_name from students where permanent_address_id!= (select current_address_id from students)	student_transcripts_tracking
select addresses.address_id, addresses.line_1 from addresses join students on addresses.address_id = students.current_address_id group by addresses.address_id order by count(*) desc limit 1	student_transcripts_tracking
select addresses.address_id, addresses.line_1, addresses.line_2 from addresses join students on addresses.address_id = students.current_address_id group by addresses.address_id order by count(*) desc limit 1	student_transcripts_tracking
select avg(transcript_date) from transcripts	student_transcripts_tracking
select avg(transcript_date) from transcripts	student_transcripts_tracking
select transcript_date, other_details from transcripts order by transcript_date asc limit 1	student_transcripts_tracking
select transcript_date, other_details from transcripts order by transcript_date asc limit 1	student_transcripts_tracking
select count(*) from transcripts	student_transcripts_tracking
select count(*) from transcripts	student_transcripts_tracking
select transcript_date from transcripts order by transcript_date desc limit 1	student_transcripts_tracking
select max(transcript_date) from transcripts	student_transcripts_tracking
select count(*), student_enrolment.student_enrolment_id from student_enrolment join transcript_contents on student_enrolment.student_enrolment_id = transcript_contents.student_course_id group by student_enrolment.student_enrolment_id order by count(*) desc limit 1	student_transcripts_tracking
select courses.course_name, count(*) from courses join student_enrolment_courses on courses.course_id = student_enrolment_courses.course_id group by courses.course_name	student_transcripts_tracking
select transcripts.transcript_date, transcripts.transcript_id from transcripts join transcript_contents on transcripts.transcript_id = transcript_contents.transcript_id group by transcripts.transcript_id order by count(*) asc limit 1	student_transcripts_tracking
select transcripts.transcript_date, transcripts.transcript_id from transcripts join transcript_contents on transcripts.transcript_id = transcript_contents.transcript_id group by transcripts.transcript_id order by count(*) asc limit 1	student_transcripts_tracking
select semester_id from student_enrolment join degree_programs on student_enrolment.degree_program_id = degree_programs.degree_program_id where degree_programs.degree_summary_name = 'Master' intersect select semester_id from student_enrolment join degree_programs on student_enrolment.degree_program_id = degree_programs.degree_program_id where degree_programs.degree_summary_name = 'Bachelor'	student_transcripts_tracking
select semester_id from student_enrolment where student_id = (select student_id from student_enrolment where degree_program_id = 'MA' intersect select semester_id from student_enrolment where degree_program_id = 'B')	student_transcripts_tracking
select count(distinct current_address_id) from students	student_transcripts_tracking
select distinct addresses.address_id from addresses join students on addresses.address_id = students.current_address_id	student_transcripts_tracking
select other_student_details from students order by other_student_details desc	student_transcripts_tracking
select other_student_details from students order by other_student_details desc	student_transcripts_tracking
select section_description from sections where section_name = "H"	student_transcripts_tracking
select section_description from sections where section_name = "H"	student_transcripts_tracking
select first_name from students where permanent_address_id in (select address_id from addresses where country = 'Haiti') or cell_mobile_number = "09700166582"	student_transcripts_tracking
select distinct students.first_name from addresses join students on addresses.address_id = students.permanent_address_id where addresses.country = "Haiti" or students.cell_mobile_number = "09700166582"	student_transcripts_tracking
select title from cartoon order by title	tvshow
select title from cartoon order by title	tvshow
select title from cartoon where directed_by = "Ben Jones"	tvshow
select title from cartoon where directed_by = "Ben Jones"	tvshow
select count(*) from cartoon where written_by = "Joseph Kuhr"	tvshow
select count(*) from cartoon where written_by = "Joseph Kuhr"	tvshow
select title, directed_by from cartoon order by original_air_date	tvshow
select title, directed_by from cartoon order by original_air_date	tvshow
select title from cartoon where directed_by = "Ben Jones" or directed_by = "Brandon Vietti"	tvshow
select title from cartoon where directed_by = "Ben Jones" or directed_by = "Brandon Vietti"	tvshow
select country, count(*) from tv_channel group by country order by count(*) desc limit 1	tvshow
select country, count(*) from tv_channel group by country order by count(*) desc limit 1	tvshow
select count(distinct series_name), content from tv_channel	tvshow
select count(distinct series_name), count(distinct content) from tv_channel	tvshow
select content from tv_channel where series_name = "Sky Radio"	tvshow
select content from tv_channel where series_name = "Sky Radio"	tvshow
select package_option from tv_channel where series_name = "Sky Radio"	tvshow
select package_option from tv_channel where series_name = "Sky Radio"	tvshow
select count(*) from tv_channel where language = "English"	tvshow
select count(*) from tv_channel where language = "English"	tvshow
select language, count(*) from tv_channel group by language order by count(*) asc limit 1	tvshow
select language, count(*) from tv_channel group by language order by count(*) asc limit 1	tvshow
select language, count(*) from tv_channel group by language	tvshow
select language, count(*) from tv_channel group by language	tvshow
select tv_channel.series_name from cartoon join tv_channel on cartoon.channel = tv_channel.id where cartoon.title = "The Rise of the Blue Beetle! "	tvshow
select tv_channel.series_name from cartoon join tv_channel on cartoon.channel = tv_channel.id where cartoon.title = "The Rise of the Blue Beetle!"	tvshow
select cartoon.title from cartoon join tv_channel on cartoon.channel = tv_channel.id where tv_channel.series_name = "Sky Radio"	tvshow
select cartoon.title from cartoon join tv_channel on cartoon.channel = tv_channel.id where tv_channel.series_name = "Sky Radio"	tvshow
select episode from tv_series order by rating	tvshow
select episode from tv_series order by rating	tvshow
select episode, rating from tv_series order by rating desc limit 3	tvshow
select episode, rating from tv_series order by rating desc limit 3	tvshow
select min(share), max(share) from tv_series	tvshow
select max(share), min(share) from tv_series	tvshow
select air_date from tv_series where episode = "A love of a Lifetime"	tvshow
select air_date from tv_series where episode = "A love of a Lifetime"	tvshow
select weekly_rank from tv_series where episode = "A love of a Lifetime"	tvshow
select weekly_rank from tv_series where episode = "A love of a Lifetime"	tvshow
select tv_channel.series_name from tv_channel join tv_series on tv_channel.id = tv_series.channel where tv_series.episode = "A love of a Lifetime"	tvshow
select tv_channel.series_name from tv_series join tv_channel on tv_series.id = tv_channel.id where tv_series.episode = "A love of a Lifetime"	tvshow
select tv_series.episode from tv_series join tv_channel on tv_series.channel = tv_channel.id where tv_channel.series_name = "Sky Radio"	tvshow
select tv_series.episode from tv_series join tv_channel on tv_series.channel = tv_channel.id where tv_channel.series_name = "Sky Radio"	tvshow
select directed_by, count(*) from cartoon group by directed_by	tvshow
select directed_by, count(*) from cartoon group by directed_by	tvshow
select production_code, channel from cartoon order by original_air_date desc limit 1	tvshow
select production_code, channel from cartoon order by original_air_date desc limit 1	tvshow
select package_option, series_name from tv_channel where hight_definition_tv = 1	tvshow
select package_option, series_name from tv_channel where hight_definition_tv = "Hight_definition_TV"	tvshow
select tv_channel.country from cartoon join tv_channel on cartoon.channel = tv_channel.id where cartoon.written_by = 'Todd Casey'	tvshow
select tv_channel.country from cartoon join tv_channel on cartoon.id = tv_channel.id where cartoon.written_by = 'Todd Casey'	tvshow
select country from tv_channel except select tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.written_by = 'Todd Casey'	tvshow
select country from tv_channel except select tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.written_by = "Todd Casey"	tvshow
select tv_channel.series_name, tv_channel.country from cartoon join tv_channel on cartoon.channel = tv_channel.id where cartoon.directed_by = "Ben Jones" intersect select tv_channel.series_name, tv_channel.country from cartoon join tv_channel on cartoon.channel = tv_channel.id where cartoon.directed_by = "Michael Chang"	tvshow
select tv_channel.series_name, tv_channel.country from cartoon join tv_channel on cartoon.channel = tv_channel.id where cartoon.directed_by = "Ben Jones" intersect select tv_channel.series_name, tv_channel.country from cartoon join tv_channel on cartoon.channel = tv_channel.id where cartoon.directed_by = "Michael Chang"	tvshow
select pixel_aspect_ratio_par, country from tv_channel where language!= 'English'	tvshow
select pixel_aspect_ratio_par, country from tv_channel where language!= 'English'	tvshow
select id from tv_channel where country > 2	tvshow
select id from tv_channel group by id having count(*) > 2	tvshow
select id from tv_channel except select channel from cartoon where directed_by = "Ben Jones"	tvshow
select id from tv_channel except select channel from cartoon where directed_by = "Ben Jones"	tvshow
select package_option from tv_channel except select tv_channel.package_option from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.directed_by = "Ben Jones"	tvshow
select package_option from tv_channel except select tv_channel.package_option from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.directed_by = "Ben Jones"	tvshow
select count(*) from poker_player	poker_player
select count(*) from poker_player	poker_player
select earnings from poker_player order by earnings desc	poker_player
select earnings from poker_player order by earnings desc	poker_player
select final_table_made, best_finish from poker_player	poker_player
select final_table_made, best_finish from poker_player	poker_player
select avg(earnings) from poker_player	poker_player
select avg(earnings) from poker_player	poker_player
select money_rank from poker_player order by earnings desc limit 1	poker_player
select money_rank from poker_player order by earnings desc limit 1	poker_player
select max(final_table_made) from poker_player where earnings < 200000	poker_player
select max(final_table_made) from poker_player where earnings < 200000	poker_player
select people.name from poker_player join people on poker_player.people_id = people.people_id	poker_player
select people.name from poker_player join people on poker_player.people_id = people.people_id	poker_player
select people.name from poker_player join people on poker_player.people_id = people.people_id where poker_player.earnings > 300000	poker_player
select people.name from poker_player join people on poker_player.people_id = people.people_id where poker_player.earnings > 300000	poker_player
select people.name from poker_player join people on poker_player.people_id = people.people_id order by poker_player.final_table_made asc	poker_player
select people.name from poker_player join people on poker_player.people_id = people.people_id order by poker_player.final_table_made asc	poker_player
select people.birth_date from poker_player join people on poker_player.people_id = people.people_id order by poker_player.earnings limit 1	poker_player
select people.birth_date from poker_player join people on poker_player.people_id = people.people_id order by poker_player.earnings limit 1	poker_player
select poker_player.money_rank from poker_player join people on poker_player.people_id = people.people_id order by people.height desc limit 1	poker_player
select poker_player.money_rank from poker_player join people on poker_player.people_id = people.people_id order by people.height desc limit 1	poker_player
select avg(poker_player.earnings) from poker_player join people on poker_player.people_id = people.people_id where people.height > 200	poker_player
select avg(poker_player.earnings) from poker_player join people on poker_player.people_id = people.people_id where people.height > 200	poker_player
select people.name from poker_player join people on poker_player.people_id = people.people_id order by poker_player.earnings desc	poker_player
select people.name from poker_player join people on poker_player.people_id = people.people_id order by poker_player.earnings desc	poker_player
select nationality, count(*) from people group by nationality	poker_player
select nationality, count(*) from people group by nationality	poker_player
select nationality from people group by nationality order by count(*) desc limit 1	poker_player
select nationality from people group by nationality order by count(*) desc limit 1	poker_player
select nationality from people group by nationality having count(*) >= 2	poker_player
select nationality from people group by nationality having count(*) >= 2	poker_player
select name, birth_date from people order by name asc	poker_player
select name, birth_date from people order by name	poker_player
select name from people where nationality!= "Russia"	poker_player
select name from people where nationality!= "Russia"	poker_player
select name from people where people_id not in (select people_id from poker_player)	poker_player
select name from people where people_id not in (select people_id from poker_player)	poker_player
select count(distinct nationality) from people	poker_player
select count(distinct nationality) from people	poker_player
select count(distinct state) from area_code_state	voter_1
select contestant_number, contestant_name from contestants order by contestant_name desc	voter_1
select vote_id, phone_number, state from votes	voter_1
select max(area_code), min(area_code) from area_code_state	voter_1
select max(created) from votes where state = "CA"	voter_1
select contestant_name from contestants where contestant_name!= "Jessie Alloway"	voter_1
select distinct state, created from votes	voter_1
select votes.contestant_number, contestants.contestant_name from votes join contestants on votes.contestant_number = contestants.contestant_number group by votes.contestant_number having count(*) >= 2	voter_1
select contestants.contestant_number, contestants.contestant_name from contestants join votes on contestants.contestant_number = votes.contestant_number group by contestants.contestant_number order by count(*) limit 1	voter_1
select count(*) from votes where state = "NY" or state = "CA"	voter_1
select count(*) from contestants where contestant_number not in (select contestant_number from votes)	voter_1
select area_code_state.area_code from area_code_state join votes on area_code_state.area_code = votes.state	voter_1
select votes.created, votes.state, votes.phone_number from contestants join votes on contestants.contestant_number = votes.contestant_number where contestants.contestant_name = "Tabatha Gehling"	voter_1
select area_code_state.area_code from area_code_state join votes on area_code_state.area_code = votes.state join contestants on votes.contestant_number = contestants.contestant_number where contestants.contestant_name = "Tabatha Gehling" intersect select area_code_state.area_code from area_code_state join votes on area_code_state.area_code = votes.state join contestants on votes.contestant_number = contestants.contestant_number where contestants.contestant_name = "Kelly Clauss"	voter_1
select contestants.contestant_name from contestants join area_code_state on contestants.contestant_number = area_code_state.area_code where area_code_state.state = "AL"	voter_1
select name from country where indepyear > 1950	world_1
select name from country where indepyear > 1950	world_1
select count(*) from country where governmentform = "Republic"	world_1
select count(*) from country where governmentform = "Republic"	world_1
select sum(surfacearea) from country where region = 'Caribbean'	world_1
select sum(surfacearea) from country where continent = "Carribean"	world_1
select continent from country where name = 'Anguilla'	world_1
select continent from country where name = 'Anguilla'	world_1
select country.region from city join country on city.countrycode = country.code where city.name = 'Kabul'	world_1
select country.region from city join country on city.countrycode = country.code where city.name = 'Kabul'	world_1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.name = "Aruba" group by countrylanguage.language order by count(*) desc limit 1	world_1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.name = "Aruba" group by countrylanguage.language order by count(*) desc limit 1	world_1
select population, lifeexpectancy from country where name = 'Brazil'	world_1
select population, lifeexpectancy from country where name = 'Brazil'	world_1
select region, population from country where name = 'Angola'	world_1
select region, population from country where name = 'Angola'	world_1
select avg(lifeexpectancy) from country where region = "Central Africa"	world_1
select avg(lifeexpectancy) from country where region = "Central Africa"	world_1
select name from country where continent = 'Asia' order by lifeexpectancy limit 1	world_1
select name from country where continent = 'Asia' order by lifeexpectancy limit 1	world_1
select sum(population), max(gnp) from country where continent = 'Asia'	world_1
select gnp, population from country where continent = 'Asia' order by gnp desc limit 1	world_1
select avg(lifeexpectancy) from country where continent = 'Africa' and governmentform = 'Republic'	world_1
select avg(lifeexpectancy) from country where continent = 'Africa' and governmentform = 'Republic'	world_1
select sum(surfacearea) from country where continent = 'Asia' intersect select sum(surfacearea) from country where continent = 'Europe'	world_1
select sum(surfacearea) from country where continent = 'Asia' or continent = 'Europe'	world_1
select population from city where district = 'Gelderland'	world_1
select sum(population) from city where district = 'Gelderland'	world_1
select avg(gnp), sum(population) from country where governmentform = 'US Territory'	world_1
select avg(gnp), sum(population) from country where governmentform = 'US Territory'	world_1
select count(distinct language) from countrylanguage	world_1
select count(distinct language) from countrylanguage	world_1
select count(distinct governmentform) from country where continent = 'Africa'	world_1
select count(distinct governmentform) from country where continent = 'Africa'	world_1
select count(countrylanguage.language) from country join countrylanguage on country.code = countrylanguage.countrycode where country.name = "Aruba"	world_1
select count(countrylanguage.language) from country join countrylanguage on country.code = countrylanguage.countrycode where country.name = "Aruba"	world_1
select count(distinct countrylanguage.language) from country join countrylanguage on country.code = countrylanguage.countrycode where country.name = 'Afghanistan'	world_1
select count(distinct countrylanguage.language) from country join countrylanguage on country.code = countrylanguage.countrycode where country.name = 'Afghanistan'	world_1
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode group by country.name order by count(*) desc limit 1	world_1
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode group by country.name order by count(*) desc limit 1	world_1
select country.continent from country join countrylanguage on country.code = countrylanguage.countrycode group by country.continent order by count(*) desc limit 1	world_1
select country.continent from country join countrylanguage on country.code = countrylanguage.countrycode group by country.continent order by count(*) desc limit 1	world_1
select count(*) from countrylanguage where language = 'English' intersect select count(*) from countrylanguage where language = 'Dutch'	world_1
select count(*) from countrylanguage where language = "English" intersect select count(*) from countrylanguage where language = "Dutch"	world_1
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' intersect select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'French'	world_1
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' intersect select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'French'	world_1
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' intersect select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'French'	world_1
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' intersect select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'French'	world_1
select count(distinct country.continent) from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'Chinese'	world_1
select count(distinct country.continent) from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'Chinese'	world_1
select distinct country.region from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' or countrylanguage.language = 'Dutch'	world_1
select country.region from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'Dutch' or countrylanguage.language = 'English'	world_1
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' or countrylanguage.language = 'Dutch'	world_1
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' or countrylanguage.language = 'Dutch'	world_1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.continent = 'Asia' group by countrylanguage.language order by count(*) desc limit 1	world_1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.continent = 'Asia' group by countrylanguage.language order by count(*) desc limit 1	world_1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.governmentform = "Republic" group by countrylanguage.language having count(*) = 1	world_1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.governmentform = "Republic" group by countrylanguage.language having count(*) = 1	world_1
select city.name from city join countrylanguage on city.countrycode = countrylanguage.countrycode where countrylanguage.language = 'English' order by city.population desc limit 1	world_1
select city.name from city join countrylanguage on city.countrycode = countrylanguage.countrycode where countrylanguage.language = 'English' order by city.population desc limit 1	world_1
select name, population, lifeexpectancy from country where continent = 'Asia' and surfacearea = (select max(surfacearea) from country)	world_1
select name, population, lifeexpectancy from country where continent = 'Asia' and surfacearea = (select max(surfacearea) from country where continent = 'Asia')	world_1
select avg(lifeexpectancy) from country where countrycode not in (select countrycode from countrylanguage where language =	world_1
select avg(lifeexpectancy) from country where countrycode not in (select countrycode from countrylanguage where language =	world_1
select sum(population) from country where code not in (select countrycode from countrylanguage where language = 'English')	world_1
select sum(population) from country where code not in (select countrycode from countrylanguage where language = 'English')	world_1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.headofstate = "Beatrix"	world_1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.headofstate = "Beatrix"	world_1
select count(distinct countrylanguage.language) from country join countrylanguage on country.code = countrylanguage.countrycode where country.indepyear < 1930	world_1
select count(distinct countrylanguage.language) from country join countrylanguage on country.code = countrylanguage.countrycode where country.indepyear < 1930	world_1
select name from country where surfacearea > (select max(surfacearea) from country where continent = 'Europe')	world_1
select name from country where surfacearea > (select max(surfacearea) from country where continent = 'Europe')	world_1
select name from country where population < (select min(population) from country where continent = 'Asia') and continent = 'Africa'	world_1
select name from country where population < (select min(population) from country where continent = 'Africa')	world_1
select name from country where population > (select max(population) from country where continent = 'Africa') and continent = 'Asia'	world_1
select name from country where population > (select max(population) from country where continent = 'Africa') and continent = 'Asia'	world_1
select countrycode from countrylanguage where language!= 'English'	world_1
select countrycode from countrylanguage where language!= 'English'	world_1
select countrycode from countrylanguage where language!= 'English'	world_1
select countrycode from countrylanguage where language!= 'English'	world_1
select code from country where countrylanguage.language!= "English" and country.governmentform!= "Republik"	world_1
select countrycode from countrylanguage where countrylanguage.language!= 'English' and country.governmentform!= 'Republic'	world_1
select name from city where countrycode in (select countrycode from countrylanguage where language = 'English')	world_1
select name from city where countrycode not in (select countrycode from countrylanguage where language = 'English') and continent = 'Europe'). world_	world_1
select distinct city.name from city join countrylanguage on city.countrycode = countrylanguage.countrycode where country.continent = 'Asia' and countrylanguage.language = 'Chinese' is official. world_1	world_1
select distinct city.name from country join city on country.code = city.countrycode join countrylanguage on country.code = countrylanguage.countrycode where country.continent = 'Asia' and countrylanguage.language = 'Chinese'	world_1
select name, indepyear, surfacearea from country order by population asc limit 1	world_1
select name, indepyear, surfacearea from country order by population limit 1	world_1
select population, name, headofstate from country order by surfacearea desc limit 1	world_1
select name, population, headofstate from country order by surfacearea desc limit 1	world_1
select country.name, count(*) from country join countrylanguage on country.code = countrylanguage.countrycode group by country.code having count(*) >= 3	world_1
select country.name, count(countrylanguage.language) from country join countrylanguage on country.code = countrylanguage.countrycode group by country.name having count(countrylanguage.language) > 2	world_1
select count(*), district from city where population > (select avg(population) from city) group by district	world_1
select count(*), district from city where population > (select avg(population) from city) group by district	world_1
select governmentform, sum(population) from country group by governmentform having avg(lifeexpectancy) > 72	world_1
select governmentform, sum(population) from country group by governmentform having avg(lifeexpectancy) > 72	world_1
select avg(lifeexpectancy), sum(population), continent from country where lifeexpectancy < 72 group by continent	world_1
select continent, sum(population), avg(lifeexpectancy) from country group by continent having avg(lifeexpectancy) < 72	world_1
select name, surfacearea from country order by surfacearea desc limit 5	world_1
select name, surfacearea from country order by surfacearea desc limit 5	world_1
select name from country order by population desc limit 3	world_1
select name from country order by population desc limit 3	world_1
select name from country order by population asc limit 3	world_1
select name from country order by population asc limit 3	world_1
select count(*) from country where continent = 'Asia'	world_1
select count(*) from country where continent = 'Asia'	world_1
select name from country where continent = 'Europe' and population = 80000	world_1
select name from country where continent = 'Europe' and population = 80000	world_1
select sum(population), avg(surfacearea) from country where continent = 'North America' and surfacearea > 3000	world_1
select sum(population), avg(surfacearea) from country where continent = 'North America' and surfacearea > 3000	world_1
select name from city where population between 160000 and 900000	world_1
select name from city where population between 160000 and 900000	world_1
select language from countrylanguage group by language order by count(*) desc limit 1	world_1
select language from countrylanguage group by language order by count(*) desc limit 1	world_1
select language, countrycode from countrylanguage group by countrycode order by percentage desc limit 1	world_1
select countrycode, language, percentage from countrylanguage group by countrycode order by percentage desc limit 1	world_1
select count(*) from countrylanguage where language = "Spanish" group by countrycode order by percentage desc limit 1	world_1
select count(*) from countrylanguage where language = "Spanish" and percentage = (select max(percentage) from countrylanguage where language = "Spanish")	world_1
select countrycode from countrylanguage where language = "Spanish" group by countrycode order by percentage desc limit 1	world_1
select countrycode from countrylanguage where language = "Spanish" group by countrycode having count(*) >= 2	world_1
select count(*) from conductor	orchestra
select count(*) from conductor	orchestra
select name from conductor order by age asc	orchestra
select name from conductor order by age	orchestra
select name from conductor where nationality!= "USA"	orchestra
select name from conductor where nationality!= "USA"	orchestra
select record_company from orchestra order by year_of_founded desc	orchestra
select record_company from orchestra order by year_of_founded desc	orchestra
select avg(attendance) from show	orchestra
select avg(attendance) from show	orchestra
select max(share), min(share) from performance where type!= "Live final"	orchestra
select max(share), min(share) from performance where type!= "Live final"	orchestra
select count(distinct nationality) from conductor	orchestra
select count(distinct nationality) from conductor	orchestra
select name from conductor order by year_of_work desc	orchestra
select name from conductor order by year_of_work desc	orchestra
select name from conductor order by year_of_work desc limit 1	orchestra
select name from conductor order by year_of_work desc limit 1	orchestra
select conductor.name, orchestra.orchestra from performance join orchestra on performance.orchestra_id = orchestra.orchestra_id join conductor on conductor.conductor_id = orchestra.conductor_id	orchestra
select conductor.name, orchestra.orchestra from orchestra join conductor on orchestra.conductor_id = conductor.conductor_id	orchestra
select conductor.name from conductor join orchestra on conductor.conductor_id = orchestra.conductor_id group by conductor.conductor_id having count(*) > 1	orchestra
select conductor.name from conductor join orchestra on conductor.conductor_id = orchestra.conductor_id group by conductor.name having count(*) > 1	orchestra
select conductor.name from conductor join orchestra on conductor.conductor_id = orchestra.conductor_id group by conductor.name order by count(*) desc limit 1	orchestra
select conductor.name from conductor join orchestra on conductor.conductor_id = orchestra.conductor_id group by conductor.name order by count(*) desc limit 1	orchestra
select conductor.name from conductor join orchestra on conductor.conductor_id = orchestra.conductor_id where orchestra.year_of_founded > 2008	orchestra
select conductor.name from conductor join orchestra on conductor.conductor_id = orchestra.conductor_id where orchestra.year_of_founded > 2008	orchestra
select record_company, count(*) from orchestra group by record_company	orchestra
select record_company, count(*) from orchestra group by record_company	orchestra
select major_record_format from orchestra group by major_record_format order by count(*) asc	orchestra
select major_record_format from orchestra group by major_record_format order by count(*) asc	orchestra
select record_company from orchestra group by record_company order by count(*) desc limit 1	orchestra
select record_company from orchestra group by record_company order by count(*) desc limit 1	orchestra
select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)	orchestra
select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)	orchestra
select record_company from orchestra where year_of_founded < 2003 intersect select record_company from orchestra where year_of_founded > 2003	orchestra
select record_company from orchestra where year_of_founded < 2003 intersect select record_company from orchestra where year_of_founded > 2003	orchestra
select count(*) from orchestra where major_record_format = "CD" or major_record_format = "DVD"	orchestra
select count(*) from orchestra where major_record_format = "CD" or major_record_format = "DVD"	orchestra
select orchestra.year_of_founded from performance join orchestra on performance.orchestra_id = orchestra.orchestra_id group by performance.orchestra_id having count(*) > 1	orchestra
select orchestra.year_of_founded from performance join orchestra on performance.orchestra_id = orchestra.orchestra_id group by performance.orchestra_id having count(*) > 1	orchestra
select count(*) from highschooler	network_1
select count(*) from highschooler	network_1
select name, grade from highschooler	network_1
select name, grade from highschooler	network_1
select distinct grade from highschooler	network_1
select distinct grade from highschooler	network_1
select grade from highschooler where name = 'Kyle'	network_1
select grade from highschooler where name = 'Kyle'	network_1
select name from highschooler where grade = 10	network_1
select name from highschooler where grade = 10	network_1
select id from highschooler where name = 'Kyle'	network_1
select id from highschooler where name = 'Kyle'	network_1
select count(*) from highschooler where grade = 9 or grade = 10	network_1
select count(*) from highschooler where grade = 9 or grade = 10	network_1
select count(*), grade from highschooler group by grade	network_1
select count(*), grade from highschooler group by grade	network_1
select grade from highschooler group by grade order by count(*) desc limit 1	network_1
select grade from highschooler group by grade order by count(*) desc limit 1	network_1
select grade from highschooler group by grade having count(*) >= 4	network_1
select grade from highschooler group by grade having count(*) >= 4	network_1
select student_id, count(*) from friend group by student_id	network_1
select count(*) from friend group by student_id	network_1
select highschooler.name, count(*) from highschooler join friend on highschooler.id = friend.student_id group by highschooler.id	network_1
select highschooler.name, count(*) from highschooler join friend on highschooler.id = friend.student_id group by highschooler.id	network_1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id group by friend.student_id order by count(*) desc limit 1	network_1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id group by friend.student_id order by count(*) desc limit 1	network_1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id group by friend.student_id having count(*) >= 3	network_1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id group by friend.student_id having count(*) >= 3	network_1
select friend.friend_id from highschooler join friend on highschooler.id = friend.friend_id where highschooler.name = 'Kyle'	network_1
select friend.friend_id from highschooler join friend on highschooler.id = friend.student_id where highschooler.name = 'Kyle'	network_1
select count(*) from highschooler join friend on highschooler.id = friend.student_id where highschooler.name = 'Kyle'	network_1
select count(*) from highschooler join friend on highschooler.id = friend.student_id where highschooler.name = 'Kyle'	network_1
select	network_1
select id from highschooler except select student_id from friend	network_1
select name from highschooler where id not in (select student_id from friend)	network_1
select name from highschooler where id not in (select student_id from friend)	network_1
select student_id from friend intersect select student_id from likes	network_1
select student_id from friend intersect select student_id from likes	network_1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id intersect select highschooler.name from highschooler join likes on highschooler.id = likes.student_id	network_1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id intersect select highschooler.name from highschooler join likes on highschooler.id = likes.student_id	network_1
select count(*), student_id from likes group by student_id	network_1
select student_id, count(*) from likes group by student_id	network_1
select highschooler.name, count(*) from highschooler join likes on highschooler.id = likes.student_id group by highschooler.id	network_1
select highschooler.name, count(*) from highschooler join likes on highschooler.id = likes.student_id group by highschooler.id	network_1
select highschooler.name from highschooler join likes on highschooler.id = likes.student_id group by highschooler.id order by count(*) desc limit 1	network_1
select highschooler.name from highschooler join likes on highschooler.id = likes.student_id group by likes.student_id order by count(*) desc limit 1	network_1
select highschooler.name from highschooler join likes on highschooler.id = likes.student_id group by likes.student_id having count(*) >= 2	network_1
select highschooler.name from highschooler join likes on highschooler.id = likes.student_id group by likes.student_id having count(*) >= 2	network_1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id where highschooler.grade > 5 group by highschooler.id having count(*) >= 2	network_1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id where highschooler.grade > 5 group by highschooler.id having count(*) >= 2	network_1
select count(*) from highschooler join likes on highschooler.id = likes.student_id where highschooler.name = 'Kyle'	network_1
select count(*) from highschooler join likes on highschooler.id = likes.student_id where highschooler.name = 'Kyle'	network_1
select avg(grade) from highschooler where id in (select student_id from friend)	network_1
select avg(highschooler.grade) from highschooler join friend on highschooler.id = friend.student_id	network_1
select min(grade) from highschooler where id not in (select student_id from friend)	network_1
select min(grade) from highschooler where id not in (select student_id from friend)	network_1
select state from owners intersect select state from professionals	dog_kennels
select state from owners intersect select state from professionals	dog_kennels
select avg(age) from dogs where dog_id in (select dog_id from treatments)	dog_kennels
select avg(age) from dogs where dog_id in (select dog_id from treatments)	dog_kennels
select professionals.professional_id, professionals.last_name, professionals.cell_number from treatments join professionals on treatments.professional_id = professionals.professional_id where professionals.state = "Indiana" group by professionals.professional_id having count(*) > 2	dog_kennels
select professionals.professional_id, professionals.last_name, professionals.cell_number from treatments join professionals on treatments.professional_id = professionals.professional_id where professionals.state = "Indiana" union select professionals.professional_id, professionals.last_name, professionals.cell_number from treatments join professionals on treatments.professional_id = professionals.professional_id group by professionals.professional_id having count(*) > 2	dog_kennels
select name from dogs except select dogs.name from dogs join treatments on dogs.dog_id = treatments.dog_id where treatments.cost_of_treatment > 1000	dog_kennels
select name from dogs where owner_id not in (select dog_id from treatments group by dog_id having sum(cost_of_treatment) > 1000)	dog_kennels
select first_name from owners union select first_name from professionals	dog_kennels
select first_name from owners union select first_name from professionals	dog_kennels
select professional_id, role_code, email_address from professionals except select professionals.professional_id, professionals.role_code, professionals.email_address from professionals join treatments on professionals.professional_id = treatments.professional_id	dog_kennels
select professional_id, role_code, email_address from professionals except select professionals.professional_id, professionals.role_code, professionals.email_address from professionals join treatments on professionals.professional_id = treatments.professional_id	dog_kennels
select owners.owner_id, owners.first_name, owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id group by owners.owner_id order by count(*) desc limit 1	dog_kennels
select owners.owner_id, owners.first_name, owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id group by owners.owner_id order by count(*) desc limit 1	dog_kennels
select professionals.professional_id, professionals.role_code, professionals.first_name from treatments join professionals on treatments.professional_id = professionals.professional_id group by professionals.professional_id having count(*) >= 2	dog_kennels
select professionals.professional_id, professionals.role_code, professionals.first_name from treatments join professionals on treatments.professional_id = professionals.professional_id group by professionals.professional_id having count(*) >= 2	dog_kennels
select breeds.breed_name from breeds join dogs on breeds.breed_code = dogs.breed_code group by breeds.breed_code order by count(*) desc limit 1	dog_kennels
select breeds.breed_name from breeds join dogs on breeds.breed_code = dogs.breed_code group by breeds.breed_name order by count(*) desc limit 1	dog_kennels
select owners.owner_id, owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id join treatments on dogs.dog_id = treatments.dog_id group by owners.owner_id order by sum(treatments.cost_of_treatment) desc limit 1	dog_kennels
select owners.owner_id, owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id join treatments on dogs.dog_id = treatments.dog_id group by owners.owner_id order by sum(treatments.cost_of_treatment) desc limit 1	dog_kennels
select treatment_types.treatment_type_description from treatments join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code group by treatments.treatment_type_code order by sum(treatments.cost_of_treatment) asc limit 1	dog_kennels
select treatment_types.treatment_type_description from treatments join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code group by treatments.treatment_type_code order by sum(treatments.cost_of_treatment) limit 1	dog_kennels
error
error
select treatments.professional_id, professionals.cell_number from treatments join professionals on treatments.professional_id = professionals.professional_id group by treatments.professional_id having count(*) >= 2	dog_kennels
select treatments.professional_id, professionals.cell_number from treatments join professionals on treatments.professional_id = professionals.professional_id group by treatments.professional_id having count(*) >= 2	dog_kennels
select professionals.first_name, professionals.last_name from treatments join professionals on treatments.professional_id = professionals.professional_id where treatments.cost_of_treatment < (select avg(cost_of_treatment) from treatments)	dog_kennels
select professionals.first_name, professionals.last_name from treatments join professionals on treatments.professional_id = professionals.professional_id where treatments.cost_of_treatment < (select avg(cost_of_treatment) from treatments)	dog_kennels
select treatments.date_of_treatment, professionals.first_name from treatments join professionals on treatments.professional_id = professionals.professional_id	dog_kennels
select treatments.date_of_treatment, professionals.first_name from treatments join professionals on treatments.professional_id = professionals.professional_id	dog_kennels
select treatments.cost_of_treatment, treatment_types.treatment_type_description from treatments join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code	dog_kennels
select treatments.cost_of_treatment, treatment_types.treatment_type_description from treatments join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code	dog_kennels
select owners.first_name, owners.last_name, dogs.size_code from dogs join owners on dogs.owner_id = owners.owner_id	dog_kennels
select owners.first_name, owners.last_name, dogs.size_code from owners join dogs on owners.owner_id = dogs.owner_id	dog_kennels
select owners.first_name, dogs.name from owners join dogs on owners.owner_id = dogs.owner_id	dog_kennels
select owners.first_name, dogs.name from owners join dogs on owners.owner_id = dogs.owner_id	dog_kennels
select dogs.name, treatments.date_of_treatment from treatments join dogs on treatments.dog_id = dogs.dog_id group by dogs.breed_code order by count(*) asc limit 1	dog_kennels
select dogs.name, treatments.date_of_treatment from treatments join dogs on treatments.dog_id = dogs.dog_id group by dogs.breed_code order by count(*) asc limit 1	dog_kennels
select dogs.name, owners.first_name from owners join dogs on owners.owner_id = dogs.owner_id where owners.state = "Virginia"	dog_kennels
select owners.first_name, dogs.name from owners join dogs on owners.owner_id = dogs.owner_id where owners.state = "Virginia"	dog_kennels
select date_arrived, date_departed from dogs where dog_id in (select dog_id from treatments)	dog_kennels
select date_arrived, date_departed from dogs where dog_id in (select dog_id from treatments)	dog_kennels
select owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id order by dogs.age asc limit 1	dog_kennels
select owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id order by dogs.age limit 1	dog_kennels
select email_address from professionals where state = "Hawaii" or state = "Wisconsin"	dog_kennels
select email_address from professionals where state = "Hawaii" or state = "Wisconsin"	dog_kennels
select date_arrived, date_departed from dogs	dog_kennels
select date_arrived, date_departed from dogs	dog_kennels
select count(*) from treatments	dog_kennels
select count(distinct dog_id) from treatments	dog_kennels
select count(distinct professional_id) from treatments	dog_kennels
select count(distinct professional_id) from treatments	dog_kennels
select role_code, street, city, state from professionals where city like "%west%"	dog_kennels
select role_code, street, city, state from professionals where city like "%west%"	dog_kennels
select first_name, last_name, email_address from owners where state like "%north%"	dog_kennels
select first_name, last_name, email_address from owners where state like "%north%"	dog_kennels
select count(*) from dogs where age < (select avg(age) from dogs)	dog_kennels
select count(*) from dogs where age < (select avg(age) from dogs)	dog_kennels
select cost_of_treatment from treatments order by date_of_treatment desc limit 1	dog_kennels
select cost_of_treatment from treatments order by date_of_treatment desc limit 1	dog_kennels
select count(*) from dogs where dog_id not in ( select dog_id from treatments )	dog_kennels
select count(*) from dogs where dog_id not in (select dog_id from treatments)	dog_kennels
select count(*) from owners where owner_id not in (select owner_id from dogs)	dog_kennels
select count(*) from owners where owner_id not in (select owner_id from dogs)	dog_kennels
select count(*) from professionals where professional_id not in (select professional_id from treatments)	dog_kennels
select count(*) from professionals where professional_id not in (select professional_id from treatments)	dog_kennels
select name, age, weight from dogs where abandoned_yn = 1 and abandoned_yn = 0	dog_kennels
select name, age, weight from dogs where abandoned_yn = 1 and abandoned_yn = 0	dog_kennels
select avg(age) from dogs	dog_kennels
select avg(age) from dogs	dog_kennels
select max(age) from dogs	dog_kennels
select max(age) from dogs	dog_kennels
select charge_type, charge_amount from charges group by charge_type	dog_kennels
select charge_type, charge_amount from charges group by charge_type	dog_kennels
select charge_amount from charges order by charge_type desc limit 1	dog_kennels
select charge_amount from charges order by charge_amount desc limit 1	dog_kennels
select email_address, cell_number, home_phone from professionals	dog_kennels
select email_address, cell_number, home_phone from professionals	dog_kennels
select distinct breed_code, sizes.size_description from breeds join sizes on breeds.breed_code = sizes.size_code	dog_kennels
select distinct breed_code, size_code from dogs	dog_kennels
select professionals.first_name, treatment_types.treatment_type_description from treatments join professionals on treatments.professional_id = professionals.professional_id join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code	dog_kennels
select professionals.first_name, treatment_types.treatment_type_description from treatments join professionals on treatments.professional_id = professionals.professional_id join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code	dog_kennels
select count(*) from singer	singer
select count(*) from singer	singer
select name from singer order by net_worth_millions asc	singer
select name from singer order by net_worth_millions asc	singer
select birth_year, citizenship from singer	singer
select birth_year, citizenship from singer	singer
select name from singer where citizenship!= "France"	singer
select name from singer where citizenship!= "France"	singer
select name from singer where birth_year = 1948 or birth_year = 1949	singer
select name from singer where birth_year = 1948 or birth_year = 1949	singer
select name from singer order by net_worth_millions desc limit 1	singer
select name from singer order by net_worth_millions desc limit 1	singer
select citizenship, count(*) from singer group by citizenship	singer
select citizenship, count(*) from singer group by citizenship	singer
select citizenship from singer group by citizenship order by count(*) desc limit 1	singer
select citizenship from singer group by citizenship order by count(*) desc limit 1	singer
select citizenship, max(net_worth_millions) from singer group by citizenship	singer
select citizenship, max(net_worth_millions) from singer group by citizenship	singer
select song.title, singer.name from singer join song on singer.singer_id = song.singer_id	singer
select song.title, singer.name from singer join song on singer.singer_id = song.singer_id	singer
select distinct singer.name from singer join song on singer.singer_id = song.singer_id where song.sales > 300000	singer
select distinct singer.name from singer join song on singer.singer_id = song.singer_id where song.sales > 300000	singer
select singer.name from singer join song on singer.singer_id = song.singer_id group by song.singer_id having count(*) > 1	singer
select singer.name from singer join song on singer.singer_id = song.singer_id group by song.singer_id having count(*) > 1	singer
select singer.name, sum(song.sales) from singer join song on singer.singer_id = song.singer_id group by singer.name	singer
select singer.name, sum(song.sales) from singer join song on singer.singer_id = song.singer_id group by singer.name	singer
select name from singer where singer_id not in (select singer_id from song)	singer
select name from singer where singer_id not in (select singer_id from song)	singer
select citizenship from singer where birth_year < 1945 intersect select citizenship from singer where birth_year > 1955	singer
select citizenship from singer where birth_year < 1945 intersect select citizenship from singer where birth_year > 1955	singer
select count(*) from other_available_features	real_estate_properties
select ref_feature_types.feature_type_name from ref_feature_types join other_available_features on ref_feature_types.feature_type_code = other_available_features.feature_type_code where other_available_features.feature_name = "AirCon"	real_estate_properties
select ref_property_types.property_type_description from ref_property_types join properties on properties.property_type_code = ref_property_types.property_type_code	real_estate_properties
select property_name from properties where property_type_code = "House" or room_count > 1	real_estate_properties
