select count(*) from stadium
select count(*) from stadium
select Age,Name,Country from singer order by Age desc 
select Age,Name,Country from singer order by Age desc 
select avg(Age),max(Age),min(Age) from singer where Country = 'terminal'
select avg(Age),max(Age),min(Age) from singer where Country = 'terminal'
select Song_release_year,Song_Name from singer order by Age asc limit 1
select Song_release_year,Song_Name from singer order by Age asc limit 1
select Country from singer where Age > 'terminal'
select Country from singer where Age > 'terminal'
select Country,count(*) from singer group by Country
select count(*),Country from singer group by Country
select Song_Name from singer where Age > (select avg(Age) from singer)
select Song_Name from singer where Age > (select avg(Age) from singer)
select Location,Name from stadium where Capacity between 'terminal' and 'terminal'
select Location,Name from stadium where Capacity between 'terminal' and 'terminal'
select max(Capacity),avg(Capacity) from stadium
select max(Capacity),avg(Capacity) from stadium
select Name,Capacity from stadium order by Average desc limit 1
select Name,Capacity from stadium order by Average desc limit 1
select count(*) from concert where Year > 'terminal' and Year = 'terminal'
select count(*) from concert where Year != 'terminal' and Year = 'terminal'
select T1.Name,count(*) from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID group by T2.Stadium_ID
select T1.Name,count(*) from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID group by T2.Stadium_ID
select T1.Capacity,T1.Name from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID group by T1.Stadium_ID where T2.Year >= 'terminal' order by count(*) desc limit 1
select T1.Capacity,T1.Name from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID group by T1.Stadium_ID where T2.Year >= 'terminal' order by count(*) desc limit 1
select Year from concert group by Year order by count(*) desc limit 1
select Year from concert group by Year order by count(*) desc limit 1
select Name from stadium where Stadium_ID not in (select Stadium_ID from concert)
select Name from stadium where Stadium_ID not in (select Stadium_ID from concert)
select Country from singer where Age > 'terminal' intersect select Country from singer where Age < 'terminal'
select Name from stadium except select T1.Name from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID where T2.Year = 'terminal'
select Name from stadium except select T1.Name from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID where T2.Year = 'terminal'
select concert_Name,Theme,count(*) from concert group by concert_ID
select Theme,concert_Name,count(*) from concert group by concert_ID
select count(*),T1.Name from singer as T1 join singer_in_concert as T2 on T1.Singer_ID = T2.Singer_ID group by T2.Singer_ID
select T1.Name,count(*) from singer as T1 join singer_in_concert as T2 on T1.Singer_ID = T2.Singer_ID group by T2.Singer_ID
select T1.Name from singer as T1 join singer_in_concert as T2 on T1.Singer_ID = T2.Singer_ID join concert as T3 on T2.concert_ID = T3.concert_ID where T3.Year = 'terminal'
select T1.Name from singer as T1 join singer_in_concert as T2 on T1.Singer_ID = T2.Singer_ID join concert as T3 on T2.concert_ID = T3.concert_ID where T3.Year = 'terminal'
select Country,Name from singer where Song_Name like 'terminal'
select Country,Name from singer where Song_Name like 'terminal'
select T1.Location,T1.Name from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID where T2.Year = 'terminal' intersect select T1.Location,T1.Name from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID where T2.Year = 'terminal'
select T1.Location,T1.Name from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID where T2.Year = 'terminal' intersect select T1.Location,T1.Name from stadium as T1 join concert as T2 on T1.Stadium_ID = T2.Stadium_ID where T2.Year = 'terminal'
select count(*) from stadium order by Capacity desc limit 1
select count(*) from stadium order by Capacity desc limit 1
select count(*) from Pets where weight > 'terminal'
select count(*) from Pets where weight > 'terminal'
select weight from Pets order by pet_age asc limit 1
select weight from Pets order by pet_age asc limit 1
select PetType,max(weight) from Pets group by PetType
select PetType,max(weight) from Pets group by PetType
select count(*) from Student where Age > 'terminal'
select count(*) from Student where Age > 'terminal'
select count(*) from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal' or T1.Sex = 'terminal'
select count(*) from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T1.Sex = 'terminal' or T3.PetType = 'terminal'
select count(PetType) from Pets
select count(PetType) from Pets
select T1.Fname from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType <= 'terminal' and T3.PetType = 'terminal'
select T1.Fname from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType <= 'terminal' and T3.PetType = 'terminal'
select T1.Fname from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal' intersect select T1.Fname from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal'
select T1.Fname from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal' intersect select T1.Fname from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal'
select Major,Age from Student where StuID not in (select T1.StuID from Has_Pet as T1 join Pets as T2 on T1.PetID = T2.PetID where T2.PetType = 'terminal')
select Age,Major from Student where StuID not in (select T1.StuID from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal')
select StuID from Student except select T1.StuID from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal'
select StuID from Student except select T1.StuID from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal'
select T1.Fname,T1.Age from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal' or T1.StuID not in (select T1.StuID from Has_Pet as T1 join Pets as T2 on T1.PetID = T2.PetID where T2.PetType = 'terminal')
select T1.Fname,T1.Age from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal' or T1.StuID not in (select T1.StuID from Has_Pet as T1 join Pets as T2 on T1.PetID = T2.PetID where T2.PetType = 'terminal')
select PetType,weight from Pets order by pet_age asc limit 1
select PetType,weight from Pets order by pet_age asc limit 1
select PetID,weight from Pets where pet_age > 'terminal'
select PetID,weight from Pets where pet_age > 'terminal'
select max(pet_age),avg(pet_age),PetType from Pets group by PetType
select max(pet_age),avg(pet_age),PetType from Pets group by PetType
select PetType,avg(weight) from Pets group by PetType
select PetType,avg(weight) from Pets group by PetType
select Fname,Age from Student
select Fname,Age from Student
select T3.PetID from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T1.LName = 'terminal'
select T3.PetID from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T1.LName = 'terminal'
select count(*),StuID from Student group by StuID
select T2.StuID,T1.StuID from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID group by T1.StuID
select T1.Sex,T1.Fname from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID group by T2.StuID having count(*) > 'terminal'
select T1.Sex,T1.Fname from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID group by T2.StuID having count(*) > 'terminal'
select T1.LName from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.pet_age = 'terminal' or T3.PetType = 'terminal'
select T1.LName from Student as T1 join Has_Pet as T2 on T1.StuID = T2.StuID join Pets as T3 on T2.PetID = T3.PetID where T3.PetType = 'terminal' or T3.pet_age = 'terminal'
select avg(Age) from Student where StuID not in (select StuID from Has_Pet)
select avg(Age) from Student where StuID not in (select StuID from Has_Pet)
select count(*) from continents
select count(*) from continents
select T1.ContId,T2.Continent,count(*) from continents as T1 join countries as T2 on T1.ContId = T2.Continent group by T2.Continent
select T1.ContId,T2.Continent,count(*) from continents as T1 join countries as T2 on T1.ContId = T2.Continent group by T2.Continent
select count(*) from continents
select count(*) from continents
select T1.FullName,count(*),T2.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker
select T1.FullName,T2.Maker,count(*) from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker
select T1.Model from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id order by T3.Horsepower asc limit 1
select T1.Model from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id order by T3.Horsepower asc limit 1
select T1.Model from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id where T3.Weight < (select avg(Weight) from cars_data)
select T1.Model from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id where T3.Weight < (select avg(Weight) from cars_data)
select T1.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker join car_names as T3 on T2.Model = T3.Model join cars_data as T4 on T3.MakeId = T4.Id where T4.Year = 'terminal'
select T1.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker join car_names as T3 on T2.Model = T3.Model join cars_data as T4 on T3.MakeId = T4.Id where T4.Year = 'terminal'
select T1.Make,T2.Year from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T2.Year = (select min(Year) from cars_data)
select T2.Year,T1.Make from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T2.Year = (select min(Year) from cars_data)
select T1.Model from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id where T3.Year > 'terminal'
select T1.Model from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id where T3.Year > 'terminal'
select Continent,count(*) from continents group by Continent
select Continent,count(*) from continents group by Continent
select T1.CountryName from countries as T1 join car_makers as T2 on T1.CountryId = T2.Country group by T2.Country order by count(*) desc limit 1
select CountryName from countries group by CountryId order by count(*) desc limit 1
select T1.FullName,T2.Maker,count(*) from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker
select T1.FullName,count(*),T2.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker
select T2.Accelerate from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T1.Make = 'terminal'
select T2.Accelerate from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T1.Make = 'terminal'
select count(*) from countries where CountryName = 'terminal'
select count(*) from countries where CountryName = 'terminal'
select count(*) from countries where CountryName = 'terminal'
select count(*) from countries where CountryName = 'terminal'
select avg(MPG) from cars_data where Cylinders = 'terminal'
select avg(MPG) from cars_data where Cylinders = 'terminal'
select Weight from cars_data where Cylinders = 'terminal' or Year = 'terminal' order by Weight asc limit 1
select Weight from cars_data where Cylinders = 'terminal' or Year = 'terminal' order by Weight asc limit 1
select Maker,Model from model_list
select Maker,Model from model_list
select CountryId,CountryName from countries group by CountryId having count(*) >= 'terminal'
select T2.Country,T1.CountryName from countries as T1 join car_makers as T2 on T1.CountryId = T2.Country group by T1.CountryId having count(*) >= 'terminal'
select count(*) from cars_data where Horsepower > 'terminal'
select count(*) from cars_data where Horsepower > 'terminal'
select avg(Weight),Year from cars_data group by Year
select avg(Weight),Year from cars_data group by Year
select T2.CountryName from continents as T1 join countries as T2 on T1.ContId = T2.Continent group by T2.CountryName where T1.Continent = 'terminal' having count(*) >= 'terminal'
select T2.CountryName from continents as T1 join countries as T2 on T1.ContId = T2.Continent group by T2.CountryName where T1.Continent = 'terminal' having count(*) >= 'terminal'
select T1.Make,T2.Horsepower from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T2.Cylinders = 'terminal' order by T2.Horsepower desc limit 1
select T1.Make,T2.Horsepower from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T2.Cylinders = 'terminal' order by T2.Horsepower desc limit 1
select T1.Model from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id order by T3.MPG desc limit 1
select T1.Model from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id order by T3.MPG desc limit 1
select avg(Horsepower) from cars_data where Year < 'terminal'
select avg(Horsepower) from cars_data where Year < 'terminal'
select avg(T2.Edispl) from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T1.Model = 'terminal'
select avg(T3.Edispl) from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id where T1.Model = 'terminal'
select max(Accelerate),Cylinders from cars_data group by Cylinders
select max(Accelerate),Cylinders from cars_data group by Cylinders
select Model from car_names group by Model order by count(*) desc limit 1
select Model from car_names group by Model order by count(*) desc limit 1
select count(*) from cars_data where Cylinders > 'terminal'
select count(*) from cars_data where Cylinders > 'terminal'
select count(*) from cars_data where Year = 'terminal'
select count(*) from cars_data where Year = 'terminal'
select count(*) from car_makers where FullName = 'terminal'
select count(*) from car_makers where FullName = 'terminal'
select T1.FullName,T2.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker having count(*) > 'terminal'
select T1.FullName,T2.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker having count(*) > 'terminal'
select T2.Model from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker join car_names as T3 on T2.Model = T3.Model join cars_data as T4 on T3.MakeId = T4.Id where T1.FullName = 'terminal' or T4.Weight > 'terminal'
select T2.Model from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker join car_names as T3 on T2.Model = T3.Model join cars_data as T4 on T3.MakeId = T4.Id where T1.FullName = 'terminal' or T4.Weight > 'terminal'
select Year from cars_data where Weight > 'terminal' and Weight < 'terminal'
select Year from cars_data where Weight < 'terminal' and Weight > 'terminal'
select Horsepower from cars_data order by Accelerate desc limit 1
select Horsepower from cars_data order by Accelerate desc limit 1
select T2.Cylinders from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T1.Model = 'terminal' order by T2.Accelerate asc limit 1
select T2.Cylinders from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T1.Model = 'terminal' order by T2.Accelerate asc limit 1
select count(*) from cars_data where Accelerate > (select Accelerate from cars_data order by Horsepower desc limit 1)
select count(*) from cars_data where Accelerate > (select Accelerate from cars_data order by Horsepower desc limit 1)
select count(*) from continents
select count(*) from continents
select count(*) from cars_data where Cylinders > 'terminal'
select count(*) from cars_data where Cylinders > 'terminal'
select T1.Model from model_list as T1 join car_names as T2 on T1.Model = T2.Model join cars_data as T3 on T2.MakeId = T3.Id where T3.Cylinders = 'terminal' order by T3.Horsepower desc limit 1
select T1.Model from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T2.Cylinders = 'terminal' order by T2.Horsepower desc limit 1
select T2.Id,T1.Make from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T2.Horsepower > (select min(Horsepower) from cars_data order by Horsepower asc limit 1) or T2.Cylinders = 'terminal'
select T2.Id,T1.Make from car_names as T1 join cars_data as T2 on T1.MakeId = T2.Id where T2.Horsepower > (select min(Horsepower) from cars_data where Cylinders = 'terminal') or T2.Cylinders = 'terminal'
select MPG from cars_data where Cylinders = 'terminal' or Year < 'terminal' order by MPG desc limit 1
select MPG from cars_data where Year < 'terminal' or Cylinders = 'terminal' order by MPG desc limit 1
select T2.Model from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker join car_names as T3 on T2.Model = T3.Model join cars_data as T4 on T3.MakeId = T4.Id where T4.Weight < 'terminal' or T1.FullName = 'terminal'
select T2.Model from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker join car_names as T3 on T2.Model = T3.Model join cars_data as T4 on T3.MakeId = T4.Id where T1.FullName != 'terminal' or T4.Weight < 'terminal'
select CountryName from countries except select CountryName from countries
select CountryName from countries except select CountryName from countries
select T1.Id,T2.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker having count(*) >= 'terminal' intersect select T1.Id,T2.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker having count(*) > 'terminal'
select T1.Id,T2.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker having count(*) >= 'terminal' intersect select T1.Id,T2.Maker from car_makers as T1 join model_list as T2 on T1.Id = T2.Maker group by T2.Maker having count(*) > 'terminal'
select CountryId,CountryName from countries group by CountryId having count(*) > 'terminal' union select T1.CountryName,T1.CountryId from countries as T1 join car_makers as T2 on T1.CountryId = T2.Country join model_list as T3 on T2.Id = T3.Maker join car_names as T4 on T3.Model = T4.Model where T4.Model = 'terminal'
select CountryId,CountryName from countries group by CountryId having count(*) > 'terminal' union select T1.CountryId,T1.CountryName from countries as T1 join car_makers as T2 on T1.CountryId = T2.Country join model_list as T3 on T2.Id = T3.Maker where T3.Model = 'terminal'
select Country from airlines where Airline = 'terminal'
select Country from airlines where Airline = 'terminal'
select Abbreviation from airlines where Airline = 'terminal'
select Abbreviation from airlines where Airline = 'terminal'
select Abbreviation,Airline from airlines where Country = 'terminal'
select Abbreviation,Airline from airlines where Country = 'terminal'
select AirportCode,AirportName from airports where City = 'terminal'
select AirportCode,AirportName from airports where City = 'terminal'
select count(*) from airlines
select count(*) from airlines
select count(*) from airlines
select count(*) from airlines
select count(*) from airlines
select count(*) from airlines
select Airline from airlines where Abbreviation = 'terminal'
select Airline from airlines where Abbreviation = 'terminal'
select count(*) from airlines where Country = 'terminal'
select count(*) from airlines where Country = 'terminal'
select Country,City from airports where AirportName = 'terminal'
select Country,City from airports where AirportName = 'terminal'
select AirportName from airports where AirportCode = 'terminal'
select AirportName from airports where AirportCode = 'terminal'
select AirportName from airports where City = 'terminal'
select AirportName from airports where City = 'terminal'
select count(*) from flights where SourceAirport = 'terminal'
select count(*) from flights where SourceAirport = 'terminal'
select count(*) from flights where DestAirport = 'terminal'
select count(*) from flights where DestAirport = 'terminal'
select count(*) from airports where City = 'terminal'
select count(*) from airports where City = 'terminal'
select count(*) from airports where City = 'terminal'
select count(*) from airports where City = 'terminal'
select count(*) from airports where City <= 'terminal' and City = 'terminal'
select count(*) from airports where City <= 'terminal' and City = 'terminal'
select count(*) from airlines where Airline = 'terminal'
select count(*) from airlines where Airline = 'terminal'
select count(*) from airlines as T1 join flights as T2 where T2.DestAirport = 'terminal' or T1.Airline = 'terminal'
select count(*) from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal' or T1.Airline = 'terminal'
select count(*) from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal' or T1.Airline = 'terminal'
select count(*) from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal' or T1.Airline = 'terminal'
select count(*) from airlines as T1 join airports as T2 where T2.City = 'terminal' or T1.Airline = 'terminal'
select count(*) from airlines as T1 join airports as T2 where T2.City = 'terminal' or T1.Airline = 'terminal'
select City from airports group by City order by count(*) desc limit 1
select City from airports group by City order by count(*) desc limit 1
select City from airports group by City order by count(*) desc limit 1
select City from airports group by City order by count(*) desc limit 1
select DestAirport from flights group by DestAirport order by count(*) desc limit 1
select DestAirport from flights group by DestAirport order by count(*) desc limit 1
select DestAirport from flights group by DestAirport order by count(*) asc limit 1
select DestAirport from flights group by DestAirport order by count(*) asc limit 1
select Airline from airlines group by Airline order by count(*) desc limit 1
select Airline from airlines group by Airline order by count(*) desc limit 1
select Abbreviation,Country from airlines group by Airline order by count(*) asc limit 1
select Country,Abbreviation from airlines group by Airline order by count(*) asc limit 1
select T1.Airline from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal'
select T1.Airline from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal'
select T1.Airline from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal'
select T1.Airline from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal'
select T1.Airline from airlines as T1 join flights as T2 where T2.SourceAirport = 'terminal' intersect select T1.Airline from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal'
select T1.Airline from airlines as T1 join flights as T2 where T2.SourceAirport = 'terminal' intersect select T1.Airline from airlines as T1 join flights as T2 where T2.SourceAirport = 'terminal'
select T1.Airline from airlines as T1 join flights as T2 where T2.SourceAirport = 'terminal' except select T1.Airline from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal'
select T1.Airline from airlines as T1 join flights as T2 where T2.SourceAirport = 'terminal' except select T1.Airline from airlines as T1 join airports as T2 where T2.AirportCode = 'terminal'
select Airline from airlines group by Airline having count(*) > 'terminal'
select Airline from airlines group by Airline having count(*) > 'terminal'
select Airline from airlines group by Airline having count(*) < 'terminal'
select Airline from airlines group by Airline having count(*) < 'terminal'
select T2.FlightNo from airlines as T1 join flights as T2 where T1.Airline = 'terminal'
select T2.FlightNo from airlines as T1 join flights as T2 where T1.Airline = 'terminal'
select FlightNo from flights where SourceAirport = 'terminal'
select T2.FlightNo from airports as T1 join flights as T2 on T1.AirportCode = T2.DestAirport where T1.City = 'terminal'
select FlightNo from flights where DestAirport = 'terminal'
select FlightNo from flights where DestAirport = 'terminal'
select T2.FlightNo from airports as T1 join flights as T2 on T1.AirportCode = T2.DestAirport where T1.City = 'terminal'
select T2.FlightNo from airports as T1 join flights as T2 on T1.AirportCode = T2.DestAirport where T1.City = 'terminal'
select FlightNo from flights where DestAirport = 'terminal'
select T2.FlightNo from airports as T1 join flights as T2 on T1.AirportCode = T2.DestAirport where T1.City = 'terminal'
select count(*) from airports where City <= 'terminal' and City = 'terminal'
select count(*) from airports where City <= 'terminal' and City = 'terminal'
select AirportName from airports where AirportCode not in (select SourceAirport from flights)
select AirportName from airports where AirportCode not in (select SourceAirport from flights)
select count(*) from employee
select count(*) from employee
select Name from employee order by Age asc 
select Name from employee order by Age asc 
select count(*),City from employee group by City
select count(*),City from employee group by City
select City from employee group by City where Age < 'terminal' having count(*) > 'terminal'
select City from employee group by City where Age < 'terminal' having count(*) > 'terminal'
select Location,count(*) from shop group by Location
select Location,count(*) from shop group by Location
select Manager_name,District from shop order by Number_products desc limit 1
select Manager_name,District from shop order by Number_products desc limit 1
select max(Number_products),min(Number_products) from shop
select max(Number_products),min(Number_products) from shop
select District,Location,Name from shop order by Number_products desc 
select District,Location,Name from shop order by Number_products desc 
select Name from shop where Number_products > (select avg(Number_products) from shop)
select Name from shop where Number_products > (select avg(Number_products) from shop)
select T1.Name from employee as T1 join evaluation as T2 on T1.Employee_ID = T2.Employee_ID group by T2.Employee_ID order by count(*) desc limit 1
select T1.Name from employee as T1 join evaluation as T2 on T1.Employee_ID = T2.Employee_ID group by T2.Employee_ID order by count(*) desc limit 1
select T1.Name from employee as T1 join evaluation as T2 on T1.Employee_ID = T2.Employee_ID order by T2.Bonus desc limit 1
select T1.Name from employee as T1 join evaluation as T2 on T1.Employee_ID = T2.Employee_ID order by T2.Bonus desc limit 1
select Name from employee where Employee_ID not in (select Employee_ID from evaluation)
select Name from employee where Employee_ID not in (select Employee_ID from evaluation)
select Name from shop group by Shop_ID order by count(*) desc limit 1
select Name from shop group by Shop_ID order by count(*) desc limit 1
select Name from shop where Shop_ID not in (select Shop_ID from hiring)
select Name from shop where Shop_ID not in (select Shop_ID from hiring)
select count(*),Name from shop group by Name
select count(*),Name from shop group by Name
select sum(Bonus) from evaluation
select sum(Bonus) from evaluation
select * from employee
select * from employee
select District from shop where Number_products < 'terminal' intersect select District from shop where Number_products > 'terminal'
select District from shop where Number_products < 'terminal' intersect select District from shop where Number_products > 'terminal'
select count(Location) from shop
select count(Location) from shop
select count(*) from Ref_Template_Types
select count(*) from Ref_Template_Types
select Document_Description,Document_ID,Document_Name from Documents
select Document_Description,Document_ID,Document_Name from Documents
select Document_Name,Template_ID from Documents where Document_Description like 'terminal'
select Document_Name,Template_ID from Documents where Document_Description like 'terminal'
select Document_Description,Document_ID,Template_ID from Documents where Document_Name = 'terminal'
select Document_Description,Document_ID,Template_ID from Documents where Document_Name = 'terminal'
select count(Template_ID) from Documents
select count(Template_ID) from Documents
select count(*) from Ref_Template_Types where Template_Type_Code = 'terminal'
select count(*) from Templates where Template_Type_Code = 'terminal'
select count(*),Template_ID from Documents group by Template_ID
select Template_ID,count(*) from Documents group by Template_ID
select T1.Template_Type_Code,T2.Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code group by T2.Template_ID order by count(*) desc limit 1
select T1.Template_Type_Code,T2.Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code group by T2.Template_ID order by count(*) desc limit 1
select Template_ID from Documents group by Template_ID having count(*) > 'terminal'
select Template_ID from Documents group by Template_ID having count(*) > 'terminal'
select Template_ID from Templates except select Template_ID from Documents
select Template_ID from Templates except select Template_ID from Documents
select count(*) from Ref_Template_Types
select count(*) from Ref_Template_Types
select Template_ID,Version_Number,Template_Type_Code from Templates
select Template_ID,Version_Number,Template_Type_Code from Templates
select Template_Type_Code from Templates
select Template_Type_Code from Templates
select Template_ID from Templates where Template_Type_Code > 'terminal' and Template_Type_Code = 'terminal'
select Template_ID from Templates where Template_Type_Code > 'terminal' and Template_Type_Code = 'terminal'
select count(*) from Templates where Template_Type_Code = 'terminal'
select count(*) from Templates where Template_Type_Code = 'terminal'
select Version_Number,Template_Type_Code from Templates where Version_Number > 'terminal'
select Version_Number,Template_Type_Code from Templates where Version_Number > 'terminal'
select count(*),Template_Type_Code from Templates group by Template_Type_Code
select count(*),Template_Type_Code from Templates group by Template_Type_Code
select Template_Type_Code from Templates group by Template_Type_Code order by count(*) desc limit 1
select Template_Type_Code from Templates group by Template_Type_Code order by count(*) desc limit 1
select Template_Type_Code from Templates group by Template_Type_Code having count(*) < 'terminal'
select Template_Type_Code from Templates group by Template_Type_Code having count(*) < 'terminal'
select min(Version_Number),Template_Type_Code from Templates
select min(Version_Number),Template_Type_Code from Templates
select T1.Template_Type_Code from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID where T2.Document_Name = 'terminal'
select T1.Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID where T3.Document_Name = 'terminal'
select T2.Document_Name from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID where T1.Template_Type_Code = 'terminal'
select T3.Document_Name from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID where T1.Template_Type_Code = 'terminal'
select Template_Type_Code,count(*) from Templates group by Template_Type_Code
select Template_Type_Code,count(*) from Templates group by Template_Type_Code
select T2.Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code group by T1.Template_Type_Code order by count(*) desc limit 1
select Template_Type_Code from Ref_Template_Types group by Template_Type_Code order by count(*) desc limit 1
select Template_Type_Code from Templates except select Template_Type_Code from Templates
select Template_Type_Code from Templates except select Template_Type_Code from Templates
select Template_Type_Description,Template_Type_Code from Ref_Template_Types
select Template_Type_Code,Template_Type_Description from Ref_Template_Types
select Template_Type_Description from Ref_Template_Types where Template_Type_Code = 'terminal'
select Template_Type_Description from Ref_Template_Types where Template_Type_Code = 'terminal'
select Template_Type_Code from Ref_Template_Types where Template_Type_Description = 'terminal'
select Template_Type_Code from Ref_Template_Types where Template_Type_Description = 'terminal'
select Template_Type_Description from Ref_Template_Types
select Template_Type_Description from Ref_Template_Types
select T2.Template_ID from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code where T1.Template_Type_Description = 'terminal'
select T2.Template_ID from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code where T1.Template_Type_Description = 'terminal'
select count(*) from Ref_Template_Types
select count(*) from Ref_Template_Types
select count(*) from Documents where Document_Name = 'terminal'
select count(*) from Documents where Document_Name = 'terminal'
select Other_Details from Paragraphs where Paragraph_Text = 'terminal'
select Other_Details from Paragraphs where Paragraph_Text = 'terminal'
select T2.Paragraph_Text,T2.Paragraph_ID from Documents as T1 join Paragraphs as T2 on T1.Document_ID = T2.Document_ID where T1.Document_Name = 'terminal'
select T2.Paragraph_Text,T2.Paragraph_ID from Documents as T1 join Paragraphs as T2 on T1.Document_ID = T2.Document_ID where T1.Document_Name = 'terminal'
select T2.Paragraph_Text from Documents as T1 join Paragraphs as T2 on T1.Document_ID = T2.Document_ID where T1.Document_Name = 'terminal'
select T2.Paragraph_Text from Documents as T1 join Paragraphs as T2 on T1.Document_ID = T2.Document_ID where T1.Document_Name = 'terminal'
select count(*),Document_ID from Paragraphs group by Document_ID order by Document_ID asc 
select count(*),Document_ID from Paragraphs group by Document_ID order by Document_ID asc 
select T1.Document_Name,T1.Document_ID,T2.Document_ID from Documents as T1 join Paragraphs as T2 on T1.Document_ID = T2.Document_ID group by T2.Document_ID
select T1.Document_Name,T2.Document_ID,count(*) from Documents as T1 join Paragraphs as T2 on T1.Document_ID = T2.Document_ID group by T1.Document_ID
select Document_ID from Paragraphs group by Document_ID having count(*) >= 'terminal'
select Document_ID from Paragraphs group by Document_ID having count(*) >= 'terminal'
select T1.Document_Name,T2.Document_ID from Documents as T1 join Paragraphs as T2 on T1.Document_ID = T2.Document_ID group by T2.Document_ID order by count(*) desc limit 1
select T1.Document_Name,T2.Document_ID from Documents as T1 join Paragraphs as T2 on T1.Document_ID = T2.Document_ID group by T2.Document_ID order by count(*) desc limit 1
select Document_ID from Paragraphs group by Document_ID order by count(*) asc limit 1
select Document_ID from Paragraphs group by Document_ID order by count(*) asc limit 1
select Document_ID from Paragraphs group by Document_ID having count(*) between 'terminal'
select Document_ID from Paragraphs group by Document_ID having count(*) between 'terminal'
select Document_ID from Paragraphs where Paragraph_Text = 'terminal' intersect select Document_ID from Paragraphs where Paragraph_Text = 'terminal'
select Document_ID from Paragraphs where Paragraph_Text = 'terminal' intersect select Document_ID from Paragraphs where Paragraph_Text = 'terminal'
select count(*) from course
select count(*) from course
select Name from teacher order by Age asc 
select Name from teacher order by Age asc 
select Age,Hometown from teacher
select Age,Hometown from teacher
select Name from teacher where Hometown != 'terminal'
select Name from teacher where Hometown != 'terminal'
select Name from teacher where Age != 'terminal' and Age = 'terminal'
select Name from teacher where Age != 'terminal' and Age = 'terminal'
select Hometown from teacher order by Age asc limit 1
select Hometown from teacher order by Age asc limit 1
select Hometown,count(*) from teacher group by Hometown
select Hometown,count(*) from teacher group by Hometown
select Hometown from teacher group by Hometown order by count(*) desc limit 1
select Hometown from teacher group by Hometown order by count(*) desc limit 1
select Hometown from teacher group by Hometown having count(*) >= 'terminal'
select Hometown from teacher group by Hometown having count(*) >= 'terminal'
select T1.Course,T3.Name from course as T1 join course_arrange as T2 on T1.Course_ID = T2.Course_ID join teacher as T3 on T2.Teacher_ID = T3.Teacher_ID
select T1.Course,T3.Name from course as T1 join course_arrange as T2 on T1.Course_ID = T2.Course_ID join teacher as T3 on T2.Teacher_ID = T3.Teacher_ID
select T1.Course,T3.Name from course as T1 join course_arrange as T2 on T1.Course_ID = T2.Course_ID join teacher as T3 on T2.Teacher_ID = T3.Teacher_ID order by T3.Name asc 
select T1.Course,T3.Name from course as T1 join course_arrange as T2 on T1.Course_ID = T2.Course_ID join teacher as T3 on T2.Teacher_ID = T3.Teacher_ID order by T3.Name asc 
select T3.Name from course as T1 join course_arrange as T2 on T1.Course_ID = T2.Course_ID join teacher as T3 on T2.Teacher_ID = T3.Teacher_ID where T1.Course = 'terminal'
select T3.Name from course as T1 join course_arrange as T2 on T1.Course_ID = T2.Course_ID join teacher as T3 on T2.Teacher_ID = T3.Teacher_ID where T1.Course = 'terminal'
select Name,count(*) from teacher group by Name
select Name,count(*) from teacher group by Name
select Name from teacher group by Name having count(*) >= 'terminal'
select Name from teacher group by Name having count(*) >= 'terminal'
select Name from teacher where Teacher_ID not in (select Teacher_ID from course_arrange)
select Name from teacher where Teacher_ID not in (select Teacher_ID from course_arrange)
select count(*) from visitor where Age < 'terminal'
select Name from visitor where Level_of_membership > 'terminal' order by Level_of_membership desc 
select avg(Age) from visitor where Level_of_membership <= 'terminal'
select Level_of_membership,Name from visitor where Level_of_membership > 'terminal' order by Age desc 
select Museum_ID,Name from museum order by Num_of_Staff desc limit 1
select avg(Num_of_Staff) from museum where Open_Year < 'terminal'
select Open_Year,Num_of_Staff from museum where Name = 'terminal'
select Name from museum where Num_of_Staff > (select min(Num_of_Staff) from museum where Open_Year > 'terminal')
select Age,ID,Name from visitor group by ID having count(*) > 'terminal'
select T2.visitor_ID,T1.ID,T1.Name from visitor as T1 join visit as T2 on T1.ID = T2.visitor_ID group by T2.visitor_ID order by sum(T2.Total_spent) desc limit 1
select Museum_ID,Name from museum group by Museum_ID order by count(*) desc limit 1
select Name from museum where Museum_ID not in (select Museum_ID from visit)
select T1.Age,T1.Name from visitor as T1 join visit as T2 on T1.ID = T2.visitor_ID order by T2.Num_of_Ticket desc limit 1
select max(Num_of_Ticket),avg(Num_of_Ticket) from visit
select sum(T2.Total_spent) from visitor as T1 join visit as T2 on T1.ID = T2.visitor_ID where T1.Level_of_membership = 'terminal'
select T3.Name from museum as T1 join visit as T2 on T1.Museum_ID = T2.Museum_ID join visitor as T3 on T2.visitor_ID = T3.ID where T1.Open_Year < 'terminal' intersect select T3.Name from museum as T1 join visit as T2 on T1.Museum_ID = T2.Museum_ID join visitor as T3 on T2.visitor_ID = T3.ID where T1.Open_Year > 'terminal'
select count(*) from visitor where ID not in (select T2.visitor_ID from museum as T1 join visit as T2 on T1.Museum_ID = T2.Museum_ID where T1.Open_Year > 'terminal')
select count(*) from museum where Open_Year > 'terminal' and Open_Year < 'terminal'
select count(*) from players
select count(*) from players
select count(*) from players
select count(*) from players
select birth_date,first_name from players where country_code = 'terminal'
select first_name,birth_date from players where country_code = 'terminal'
select avg(winner_age),avg(loser_age) from matches
select avg(winner_age),avg(loser_age) from matches
select avg(winner_rank) from matches
select avg(winner_rank) from matches
select min(loser_rank) from matches
select min(loser_rank) from matches
select count(country_code) from players
select count(country_code) from players
select count(loser_name) from matches
select count(loser_name) from matches
select tourney_name from matches group by tourney_name having count(*) > 'terminal'
select tourney_name from matches group by tourney_name having count(*) > 'terminal'
select winner_name from matches where year = 'terminal' intersect select winner_name from matches where year = 'terminal'
select winner_name from matches where year = 'terminal' intersect select winner_name from matches where year = 'terminal'
select count(*) from matches where year <= 'terminal' and year = 'terminal'
select count(*) from matches where year != 'terminal' and year = 'terminal'
select T1.country_code,T1.first_name from players as T1 join matches as T2 on T1.player_id = T2.winner_id where T2.tourney_name = 'terminal' intersect select T1.country_code,T1.first_name from players as T1 join matches as T2 on T1.player_id = T2.winner_id where T2.tourney_name = 'terminal'
select T1.first_name,T1.country_code from players as T1 join matches as T2 on T1.player_id = T2.winner_id where T2.tourney_name = 'terminal' intersect select T1.country_code,T1.first_name from players as T1 join matches as T2 on T1.player_id = T2.winner_id where T2.tourney_name = 'terminal'
select first_name,country_code from players order by birth_date asc limit 1
select first_name,country_code from players order by birth_date asc limit 1
select first_name,last_name from players order by birth_date asc 
select first_name,last_name from players order by birth_date asc 
select last_name,first_name from players where hand = 'terminal' order by birth_date asc 
select first_name,last_name from players where hand = 'terminal' order by birth_date asc 
select T1.first_name,T1.country_code from players as T1 join rankings as T2 on T1.player_id = T2.player_id order by T2.tours desc limit 1
select T1.first_name,T1.country_code from players as T1 join rankings as T2 on T1.player_id = T2.player_id order by T2.tours desc limit 1
select year from matches group by year order by count(*) desc limit 1
select year from matches group by year order by count(*) desc limit 1
select winner_rank_points,winner_name from matches group by winner_name order by count(*) desc limit 1
select winner_rank_points,winner_name from matches group by winner_name order by count(*) desc limit 1
select winner_name from matches where tourney_name = 'terminal' order by winner_rank_points desc limit 1
select winner_name from matches where tourney_name = 'terminal' order by winner_rank_points desc limit 1
select loser_name,winner_name from matches order by minutes desc limit 1
select loser_name,winner_name from matches order by minutes desc limit 1
select avg(T2.ranking),T1.first_name from players as T1 join rankings as T2 on T1.player_id = T2.player_id group by T1.first_name
select avg(T2.ranking),T1.first_name from players as T1 join rankings as T2 on T1.player_id = T2.player_id group by T1.first_name
select sum(T2.ranking_points),T1.first_name from players as T1 join rankings as T2 on T1.player_id = T2.player_id group by T1.first_name
select sum(T2.ranking_points),T1.first_name from players as T1 join rankings as T2 on T1.player_id = T2.player_id group by T1.first_name
select count(*),country_code from players group by country_code
select count(*),country_code from players group by country_code
select country_code from players group by country_code order by count(*) desc limit 1
select country_code from players group by country_code order by count(*) desc limit 1
select country_code from players group by country_code having count(*) > 'terminal'
select country_code from players group by country_code having count(*) > 'terminal'
select sum(tours),ranking_date from rankings group by ranking_date
select sum(tours),ranking_date from rankings group by ranking_date
select count(*),year from matches group by year
select year,count(*) from matches group by year
select winner_rank,winner_name from matches order by winner_age asc limit 1
select winner_rank,winner_name from matches order by winner_age asc limit 1
select count(winner_name) from matches where winner_hand = 'terminal' or tourney_name = 'terminal'
select count(winner_name) from matches where winner_hand = 'terminal' or tourney_name = 'terminal'
select T1.birth_date,T1.first_name,T1.country_code from players as T1 join matches as T2 on T1.player_id = T2.winner_id order by T2.winner_rank_points desc limit 1
select T1.first_name,T1.birth_date,T1.country_code from players as T1 join matches as T2 on T1.player_id = T2.winner_id order by T2.winner_rank_points desc limit 1
select hand,count(*) from players group by hand
select hand,count(*) from players group by hand
select count(*) from ship where disposition_of_ship = 'terminal'
select tonnage,name from ship order by name desc 
select date,name from battle
select max(killed),min(killed) from death
select avg(injured) from death
select T2.injured,T2.killed from ship as T1 join death as T2 on T1.id = T2.caused_by_ship_id where T1.tonnage = 'terminal'
select result,name from battle where bulgarian_commander != 'terminal'
select T1.name,T1.id from battle as T1 join ship as T2 on T1.id = T2.lost_in_battle where T2.ship_type = 'terminal'
select T1.name,T1.id from battle as T1 join ship as T2 on T1.id = T2.lost_in_battle join death as T3 on T2.id = T3.caused_by_ship_id group by T1.id having sum(T3.killed) > 'terminal'
select T2.caused_by_ship_id,T1.name from ship as T1 join death as T2 on T1.id = T2.caused_by_ship_id group by T2.caused_by_ship_id order by count(*) desc limit 1
select name from battle where latin_commander = 'terminal' or bulgarian_commander = 'terminal'
select count(result) from battle
select count(*) from battle where id not in (select lost_in_battle from ship where tonnage = 'terminal')
select T1.date,T1.name from battle as T1 join ship as T2 on T1.id = T2.lost_in_battle where T2.name = 'terminal' intersect select T1.date,T1.name from battle as T1 join ship as T2 on T1.id = T2.lost_in_battle where T2.name = 'terminal'
select bulgarian_commander,result,name from battle except select T1.bulgarian_commander,T1.result,T1.name from battle as T1 join ship as T2 on T1.id = T2.lost_in_battle where T2.location = 'terminal'
select note from death where note like 'terminal'
select line_2,line_1 from Addresses
select line_2,line_1 from Addresses
select count(*) from Addresses
select count(*) from Addresses
select course_description from Courses where course_name = 'terminal'
select course_description from Courses where course_name = 'terminal'
select zip_postcode from Addresses where city = 'terminal'
select zip_postcode from Addresses where city = 'terminal'
select T1.department_id,T2.department_id from Departments as T1 join Degree_Programs as T2 on T1.department_id = T2.department_id group by T2.department_id order by count(*) desc limit 1
select T1.department_name,T2.department_id from Departments as T1 join Degree_Programs as T2 on T1.department_id = T2.department_id group by T2.department_id order by count(*) desc limit 1
select count(department_id) from Degree_Programs
select count(department_id) from Degree_Programs
select count(degree_summary_name) from Degree_Programs
select count(degree_summary_name) from Degree_Programs
select count(*) from Departments where department_name = 'terminal'
select count(*) from Departments where department_name = 'terminal'
select section_description,section_name from Sections
select section_description,section_name from Sections
select T1.course_name,T2.course_id from Courses as T1 join Sections as T2 on T1.course_id = T2.course_id group by T2.course_id having count(*) <= 'terminal'
select T1.course_name,T2.course_id from Courses as T1 join Sections as T2 on T1.course_id = T2.course_id group by T2.course_id having count(*) <= 'terminal'
select section_name from Sections order by section_name desc 
select section_name from Sections order by section_name desc 
select semester_name,semester_id from Semesters group by semester_id order by count(*) desc limit 1
select semester_id,semester_name from Semesters group by semester_id order by count(*) desc limit 1
select department_description from Departments where department_name like 'terminal'
select department_description from Departments where department_name like 'terminal'
select T1.student_id,T2.student_id,T1.first_name,T1.last_name from Students as T1 join Student_Enrolment as T2 on T1.student_id = T2.student_id group by T2.student_id having count(*) = 'terminal'
select T2.student_id,T1.middle_name,T1.first_name,T1.last_name from Students as T1 join Student_Enrolment as T2 on T1.student_id = T2.student_id group by T1.student_id having count(*) = 'terminal'
select T3.middle_name,T3.last_name,T3.first_name from Degree_Programs as T1 join Student_Enrolment as T2 on T1.degree_program_id = T2.degree_program_id join Students as T3 on T2.student_id = T3.student_id where T1.degree_summary_name = 'terminal'
select T3.middle_name,T3.first_name,T3.last_name from Degree_Programs as T1 join Student_Enrolment as T2 on T1.degree_program_id = T2.degree_program_id join Students as T3 on T2.student_id = T3.student_id where T1.degree_summary_name = 'terminal'
select degree_summary_name from Degree_Programs group by degree_summary_name order by count(*) desc limit 1
select degree_summary_name from Degree_Programs group by degree_summary_name order by count(*) desc limit 1
select T2.degree_program_id,T1.degree_program_id from Degree_Programs as T1 join Student_Enrolment as T2 on T1.degree_program_id = T2.degree_program_id group by T2.degree_program_id order by count(*) desc limit 1
select degree_summary_name,degree_program_id from Degree_Programs group by degree_program_id order by count(*) desc limit 1
select T1.student_id,T2.student_id,T1.first_name,T1.last_name from Students as T1 join Student_Enrolment as T2 on T1.student_id = T2.student_id group by T2.student_id order by count(*) desc limit 1
select T1.student_id,T2.student_id,T1.first_name,T1.last_name from Students as T1 join Student_Enrolment as T2 on T1.student_id = T2.student_id group by T2.student_id order by count(*) desc limit 1
select semester_name from Semesters where semester_id not in (select semester_id from Student_Enrolment)
select semester_name from Semesters where semester_id not in (select semester_id from Student_Enrolment)
select course_name from Courses
select course_name from Courses
select course_name from Courses group by course_name order by count(*) desc limit 1
select course_name from Courses group by course_name order by count(*) desc limit 1
select T2.last_name from Addresses as T1 join Students as T2 on T1.address_id = T2.permanent_address_id where T1.state_province_county = 'terminal' except select last_name from Students
select T2.last_name from Addresses as T1 join Students as T2 on T1.address_id = T2.permanent_address_id where T1.state_province_county = 'terminal' except select last_name from Students
select T2.transcript_id,T1.transcript_date from Transcripts as T1 join Transcript_Contents as T2 on T1.transcript_id = T2.transcript_id group by T2.transcript_id having count(*) >= 'terminal'
select T2.transcript_id,T1.transcript_date from Transcripts as T1 join Transcript_Contents as T2 on T1.transcript_id = T2.transcript_id group by T1.transcript_id having count(*) >= 'terminal'
select cell_mobile_number from Students where last_name = 'terminal' or first_name = 'terminal'
select cell_mobile_number from Students where last_name = 'terminal' or first_name = 'terminal'
select middle_name,first_name,last_name from Students order by date_first_registered asc limit 1
select middle_name,first_name,last_name from Students order by date_first_registered asc limit 1
select middle_name,first_name,last_name from Students order by date_left asc limit 1
select middle_name,first_name,last_name from Students order by date_left asc limit 1
select first_name from Students where current_address_id != 'terminal'
select first_name from Students where current_address_id != 'terminal'
select line_2,line_1,address_id from Addresses group by address_id order by count(*) desc limit 1
select line_2,address_id,line_1 from Addresses group by address_id order by count(*) desc limit 1
select avg(transcript_date) from Transcripts
select avg(transcript_date) from Transcripts
select other_details,transcript_date from Transcripts order by transcript_date asc limit 1
select other_details,transcript_date from Transcripts order by transcript_date asc limit 1
select count(*) from Addresses
select count(*) from Addresses
select transcript_date from Transcripts order by transcript_date desc limit 1
select transcript_date from Transcripts order by transcript_date desc limit 1
select count(*),student_course_id from Transcript_Contents group by student_course_id order by count(*) desc limit 1
select count(*),student_course_id from Transcript_Contents group by student_course_id order by count(*) desc limit 1
select T2.transcript_id,T1.transcript_date from Transcripts as T1 join Transcript_Contents as T2 on T1.transcript_id = T2.transcript_id group by T1.transcript_id order by count(*) asc limit 1
select T2.transcript_id,T1.transcript_date from Transcripts as T1 join Transcript_Contents as T2 on T1.transcript_id = T2.transcript_id group by T2.transcript_id order by count(*) asc limit 1
select T2.semester_id from Degree_Programs as T1 join Student_Enrolment as T2 on T1.degree_program_id = T2.degree_program_id where T1.degree_summary_name = 'terminal' intersect select T2.semester_id from Degree_Programs as T1 join Student_Enrolment as T2 on T1.degree_program_id = T2.degree_program_id where T1.degree_summary_name = 'terminal'
select T3.semester_id from Degree_Programs as T1 join Student_Enrolment as T2 on T1.degree_program_id = T2.degree_program_id join Semesters as T3 on T2.semester_id = T3.semester_id where T1.degree_summary_name = 'terminal' intersect select T3.semester_id from Degree_Programs as T1 join Student_Enrolment as T2 on T1.degree_program_id = T2.degree_program_id join Semesters as T3 on T2.semester_id = T3.semester_id where T1.degree_summary_name = 'terminal'
select count(current_address_id) from Students
select count(current_address_id) from Students
select other_student_details from Students order by other_student_details desc 
select other_student_details from Students order by other_student_details desc 
select section_description from Sections where section_name = 'terminal'
select section_description from Sections where section_name = 'terminal'
select T2.first_name from Addresses as T1 join Students as T2 on T1.address_id = T2.permanent_address_id where T1.country = 'terminal' union select first_name from Students where cell_mobile_number = 'terminal'
select T2.first_name from Addresses as T1 join Students as T2 on T1.address_id = T2.permanent_address_id where T1.country = 'terminal' union select first_name from Students where cell_mobile_number = 'terminal'
select Title from Cartoon order by Title asc 
select Title from Cartoon order by Title asc 
select Title from Cartoon where Directed_by = 'terminal'
select Title from Cartoon where Directed_by = 'terminal'
select count(*) from Cartoon where Written_by = 'terminal'
select count(*) from Cartoon where Written_by = 'terminal'
select Directed_by,Title from Cartoon order by Original_air_date asc 
select Directed_by,Title from Cartoon order by Original_air_date asc 
select Title from Cartoon where Directed_by != 'terminal' and Directed_by = 'terminal'
select Title from Cartoon where Directed_by != 'terminal' and Directed_by = 'terminal'
select count(*),Country from TV_Channel group by Country order by count(*) desc limit 1
select Country,count(*) from TV_Channel group by Country order by count(*) desc limit 1
select count(Content),count(series_name) from TV_Channel
select count(series_name),count(Content) from TV_Channel
select Content from TV_Channel where series_name = 'terminal'
select Content from TV_Channel where series_name = 'terminal'
select Package_Option from TV_Channel where series_name = 'terminal'
select Package_Option from TV_Channel where series_name = 'terminal'
select count(*) from TV_Channel where Language = 'terminal'
select count(*) from TV_Channel where Language = 'terminal'
select Language,count(*) from TV_Channel group by Language order by count(*) asc limit 1
select Language,count(*) from TV_Channel group by Language order by count(*) asc limit 1
select Language,count(*) from TV_Channel group by Language
select Language,count(*) from TV_Channel group by Language
select T1.series_name from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Title = 'terminal'
select T1.series_name from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Title = 'terminal'
select T2.Title from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T1.series_name = 'terminal'
select T2.Title from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T1.series_name = 'terminal'
select Episode from TV_series order by Rating asc 
select Episode from TV_series order by Rating asc 
select Rating,Episode from TV_series order by Rating desc limit 1
select Rating,Episode from TV_series order by Rating desc limit 1
select max(Share),min(Share) from TV_series
select max(Share),min(Share) from TV_series
select Air_Date from TV_series where Episode = 'terminal'
select Air_Date from TV_series where Episode = 'terminal'
select Weekly_Rank from TV_series where Episode = 'terminal'
select Weekly_Rank from TV_series where Episode = 'terminal'
select T1.series_name from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel where T2.Episode = 'terminal'
select T1.series_name from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel where T2.Episode = 'terminal'
select T2.Episode from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel where T1.series_name = 'terminal'
select T2.Episode from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel where T1.series_name = 'terminal'
select count(*),Directed_by from Cartoon group by Directed_by
select count(*),Directed_by from Cartoon group by Directed_by
select Channel,Production_code from Cartoon order by Original_air_date asc limit 1
select Channel,Production_code from Cartoon order by Original_air_date asc limit 1
select Package_Option,series_name from TV_Channel where Hight_definition_TV = 'terminal'
select Package_Option,series_name from TV_Channel where Hight_definition_TV = 'terminal'
select T1.Country from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Written_by = 'terminal'
select T1.Country from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Written_by = 'terminal'
select Country from TV_Channel except select T1.Country from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Written_by = 'terminal'
select Country from TV_Channel except select T1.Country from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Written_by = 'terminal'
select T1.Country,T1.series_name from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Directed_by = 'terminal' intersect select T1.Country,T1.series_name from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Directed_by = 'terminal'
select T1.series_name,T1.Country from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Directed_by = 'terminal' intersect select T1.series_name,T1.Country from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel where T2.Directed_by = 'terminal'
select Pixel_aspect_ratio_PAR,Country from TV_Channel where Language != 'terminal'
select Pixel_aspect_ratio_PAR,Country from TV_Channel where Language != 'terminal'
select id from TV_Channel group by Country having count(*) > 'terminal'
select id from TV_Channel group by Country having count(*) > 'terminal'
select id from TV_Channel except select Channel from Cartoon where Directed_by = 'terminal'
select id from TV_Channel except select Channel from Cartoon where Directed_by = 'terminal'
select Package_Option from TV_Channel where id not in (select Channel from Cartoon where Directed_by = 'terminal')
select Package_Option from TV_Channel where id not in (select Channel from Cartoon where Directed_by = 'terminal')
select count(*) from poker_player
select count(*) from poker_player
select Earnings from poker_player order by Earnings desc 
select Earnings from poker_player order by Earnings desc 
select Best_Finish,Final_Table_Made from poker_player
select Best_Finish,Final_Table_Made from poker_player
select avg(Earnings) from poker_player
select avg(Earnings) from poker_player
select Money_Rank from poker_player order by Earnings desc limit 1
select Money_Rank from poker_player order by Earnings desc limit 1
select max(Final_Table_Made) from poker_player where Earnings < 'terminal'
select max(T1.Earnings),max(T1.Final_Table_Made) from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID where T2.Height > 'terminal'
select Name from people
select Name from people
select T2.Name from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID where T1.Earnings > 'terminal'
select T2.Name from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID where T1.Earnings > 'terminal'
select T2.Name from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID order by T1.Final_Table_Made asc 
select T2.Name from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID order by T1.Final_Table_Made asc 
select T2.Birth_Date from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID order by T1.Earnings asc limit 1
select T2.Birth_Date from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID order by T1.Earnings asc limit 1
select T1.Money_Rank from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID order by T2.Height desc limit 1
select T1.Money_Rank from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID order by T2.Height desc limit 1
select avg(T1.Earnings) from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID where T2.Height > 'terminal'
select avg(T1.Earnings) from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID where T2.Height > 'terminal'
select T2.Name from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID order by T1.Earnings desc 
select T2.Name from poker_player as T1 join people as T2 on T1.People_ID = T2.People_ID order by T1.Earnings desc 
select Nationality,count(*) from people group by Nationality
select Nationality,count(*) from people group by Nationality
select Nationality from people group by Nationality order by count(*) desc limit 1
select Nationality from people group by Nationality order by count(*) desc limit 1
select Nationality from people group by Nationality having count(*) >= 'terminal'
select Nationality from people group by Nationality having count(*) >= 'terminal'
select Birth_Date,Name from people order by Name asc 
select Birth_Date,Name from people order by Name asc 
select Name from people where Nationality != 'terminal'
select Name from people where Nationality != 'terminal'
select Name from people where People_ID not in (select People_ID from poker_player)
select Name from people where People_ID not in (select People_ID from poker_player)
select count(Nationality) from people
select count(Nationality) from people
select count(*) from AREA_CODE_STATE
select contestant_number,contestant_name from CONTESTANTS order by contestant_name desc 
select vote_id,phone_number,state from VOTES
select max(area_code),min(area_code) from AREA_CODE_STATE
select max(created) from VOTES where state = 'terminal'
select contestant_name from CONTESTANTS where contestant_name != 'terminal'
select state,created from VOTES
select T1.contestant_number,T1.contestant_name from CONTESTANTS as T1 join VOTES as T2 on T1.contestant_number = T2.contestant_number group by T2.contestant_number having count(*) >= 'terminal'
select T1.contestant_number,T1.contestant_name from CONTESTANTS as T1 join VOTES as T2 on T1.contestant_number = T2.contestant_number group by T2.contestant_number order by count(*) asc limit 1
select count(*) from VOTES where state != 'terminal' and state = 'terminal'
select count(*) from CONTESTANTS where contestant_number not in (select contestant_number from VOTES)
select area_code from AREA_CODE_STATE group by area_code order by count(*) desc limit 1
select T1.state,T2.state,T2.created from AREA_CODE_STATE as T1 join VOTES as T2 on T1.state = T2.state join CONTESTANTS as T3 on T2.contestant_number = T3.contestant_number where T3.contestant_name = 'terminal'
select T1.area_code from AREA_CODE_STATE as T1 join VOTES as T2 on T1.state = T2.state join CONTESTANTS as T3 on T2.contestant_number = T3.contestant_number where T3.contestant_name = 'terminal' intersect select T1.area_code from AREA_CODE_STATE as T1 join VOTES as T2 on T1.state = T2.state join CONTESTANTS as T3 on T2.contestant_number = T3.contestant_number where T3.contestant_name = 'terminal'
select contestant_name from CONTESTANTS where contestant_name like 'terminal'
select Name from country where IndepYear > 'terminal'
select Name from country where IndepYear > 'terminal'
select count(*) from country where GovernmentForm = 'terminal'
select count(*) from country where GovernmentForm = 'terminal'
select sum(SurfaceArea) from country where Region = 'terminal'
select sum(SurfaceArea) from country where Region = 'terminal'
select Continent from country where Name = 'terminal'
select Continent from country where Name = 'terminal'
select T2.Region from city as T1 join country as T2 on T1.CountryCode = T2.Code where T1.Name = 'terminal'
select T2.Region from city as T1 join country as T2 on T1.CountryCode = T2.Code where T1.Name = 'terminal'
select T2.Language from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.Name = 'terminal' order by T2.Percentage desc limit 1
select T2.Language from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.Name = 'terminal' order by T2.Percentage desc limit 1
select Population,LifeExpectancy from country where Name = 'terminal'
select Population,LifeExpectancy from country where Name = 'terminal'
select Population,Region from country where Name = 'terminal'
select Population,Region from country where Name = 'terminal'
select avg(LifeExpectancy) from country where Region = 'terminal'
select avg(LifeExpectancy) from country where Region = 'terminal'
select Name from country where Continent = 'terminal' order by LifeExpectancy asc limit 1
select Name from country where Continent = 'terminal' order by LifeExpectancy asc limit 1
select sum(Population),max(GNP) from country where Continent = 'terminal'
select sum(Population),max(GNP) from country where Continent = 'terminal'
select avg(LifeExpectancy) from country where Continent = 'terminal' or GovernmentForm = 'terminal'
select avg(LifeExpectancy) from country where GovernmentForm = 'terminal' or Continent = 'terminal'
select sum(SurfaceArea) from country where Continent <= 'terminal' and Continent = 'terminal'
select sum(SurfaceArea) from country where Continent <= 'terminal' and Continent = 'terminal'
select sum(Population) from city where District = 'terminal'
select sum(Population) from city where District = 'terminal'
select sum(Population),avg(GNP) from country where GovernmentForm = 'terminal'
select sum(Population),avg(GNP) from country where GovernmentForm = 'terminal'
select count(Language) from countrylanguage
select count(Language) from countrylanguage
select count(GovernmentForm) from country where Continent = 'terminal'
select count(GovernmentForm) from country where Continent = 'terminal'
select count(T2.Language) from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.Name = 'terminal'
select count(T2.Language) from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.Name = 'terminal'
select count(*) from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.Name = 'terminal' or T2.IsOfficial = 'terminal'
select count(*) from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.Name = 'terminal' or T2.IsOfficial = 'terminal'
select Name from country group by Name order by count(*) desc limit 1
select Name from country group by Name order by count(*) desc limit 1
select Continent from country group by Continent order by count(*) desc limit 1
select Continent from country group by Continent order by count(*) desc limit 1
select count(*) from city
select count(*) from city
select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal' intersect select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal'
select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal' intersect select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal'
select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal' or T2.IsOfficial = 'terminal' intersect select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal' or T2.IsOfficial = 'terminal'
select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal' or T2.IsOfficial = 'terminal' intersect select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal' or T2.IsOfficial = 'terminal'
select count(T1.Continent) from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal'
select count(T1.Continent) from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal'
select T1.Region from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language != 'terminal' and T2.Language = 'terminal'
select T1.Region from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language != 'terminal' and T2.Language = 'terminal'
select * from countrylanguage where Language = 'terminal' or IsOfficial = 'terminal' union select * from countrylanguage where IsOfficial = 'terminal' or Language = 'terminal'
select * from countrylanguage where IsOfficial = 'terminal' or Language = 'terminal' union select * from countrylanguage where IsOfficial = 'terminal' or Language = 'terminal'
select T2.Language from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode group by T2.Language where T1.Continent = 'terminal' order by count(*) desc limit 1
select T2.Language from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode group by T2.Language where T1.Continent = 'terminal' order by count(*) desc limit 1
select T2.Language from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode group by T2.Language where T1.GovernmentForm = 'terminal' having count(*) = 'terminal'
select T2.Language from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode group by T2.Language where T1.GovernmentForm = 'terminal' having count(*) = 'terminal'
select T1.Population,T1.Name from city as T1 join country as T2 on T1.CountryCode = T2.Code join countrylanguage as T3 on T2.Code = T3.CountryCode where T3.Language = 'terminal' order by T1.Population desc limit 1
select T1.Population,T1.Name from city as T1 join country as T2 on T1.CountryCode = T2.Code join countrylanguage as T3 on T2.Code = T3.CountryCode where T3.Language = 'terminal' order by T1.Population desc limit 1
select Population,Name,LifeExpectancy from country where Continent = 'terminal' order by SurfaceArea desc limit 1
select Population,Name,LifeExpectancy from country where Continent = 'terminal' order by SurfaceArea desc limit 1
select avg(LifeExpectancy) from country where Name not in (select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.IsOfficial = 'terminal' or T2.Language = 'terminal')
select avg(LifeExpectancy) from country where Name not in (select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.IsOfficial = 'terminal' or T2.Language = 'terminal')
select sum(Population) from country where Name not in (select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal')
select sum(Population) from country where Name not in (select T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T2.Language = 'terminal')
select T2.Language from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.HeadOfState = 'terminal' or T2.IsOfficial = 'terminal'
select T2.Language from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.HeadOfState = 'terminal' or T2.IsOfficial = 'terminal'
select count(T2.Language) from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.IndepYear < 'terminal' or T2.IsOfficial = 'terminal'
select count(T2.Language) from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode where T1.IndepYear < 'terminal' or T2.IsOfficial = 'terminal'
select Name from country where SurfaceArea > (select min(SurfaceArea) from country where Continent = 'terminal')
select Name from country where SurfaceArea > (select min(SurfaceArea) from country where Continent = 'terminal')
select Name from country where Population > (select max(Population) from country where Continent = 'terminal') or Continent = 'terminal'
select Name from country where Population < (select sum(Population) from country where Continent = 'terminal') or Continent = 'terminal'
select Name from country where Population > (select max(Population) from country where Continent = 'terminal') or Continent = 'terminal'
select Name from country where Population > (select max(Population) from country where Continent = 'terminal') or Continent = 'terminal'
select CountryCode from countrylanguage except select CountryCode from countrylanguage where Language = 'terminal'
select CountryCode from countrylanguage except select CountryCode from countrylanguage where Language = 'terminal'
select CountryCode from countrylanguage where Language != 'terminal'
select CountryCode from countrylanguage where Language != 'terminal'
select Code from country where GovernmentForm != 'terminal' except select CountryCode from countrylanguage where Language = 'terminal'
select Code from country where GovernmentForm != 'terminal' except select CountryCode from countrylanguage where Language = 'terminal'
select T1.Name from city as T1 join country as T2 on T1.CountryCode = T2.Code where T2.Name not in (select Name from country) or T2.Continent = 'terminal'
select T1.Name from city as T1 join country as T2 on T1.CountryCode = T2.Code where T2.Name not in (select Name from country) or T2.Continent = 'terminal'
select T1.Name from city as T1 join country as T2 on T1.CountryCode = T2.Code join countrylanguage as T3 on T2.Code = T3.CountryCode where T3.IsOfficial = 'terminal' or T2.Continent = 'terminal' or T3.Language = 'terminal'
select T1.Name from city as T1 join country as T2 on T1.CountryCode = T2.Code join countrylanguage as T3 on T2.Code = T3.CountryCode where T2.Continent = 'terminal' or T3.IsOfficial = 'terminal' or T3.Language = 'terminal'
select SurfaceArea,IndepYear,Name from country order by Population asc limit 1
select SurfaceArea,IndepYear,Name from country order by Population asc limit 1
select HeadOfState,Population,Name from country order by SurfaceArea desc limit 1
select HeadOfState,Population,Name from country order by SurfaceArea desc limit 1
select count(T2.Language),T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode group by T1.Name having count(*) > 'terminal'
select count(T2.Language),T1.Name from country as T1 join countrylanguage as T2 on T1.Code = T2.CountryCode group by T1.Name having count(*) > 'terminal'
select District,count(*) from city group by District where Population > (select avg(Population) from city)
select District,count(*) from city group by District where Population > (select avg(Population) from city)
select sum(Population),GovernmentForm from country group by GovernmentForm having avg(LifeExpectancy) > 'terminal'
select sum(Population),GovernmentForm from country group by GovernmentForm having avg(LifeExpectancy) > 'terminal'
select sum(Population),avg(LifeExpectancy),Continent from country group by Continent having avg(LifeExpectancy) < 'terminal'
select sum(Population),Continent,avg(LifeExpectancy) from country group by Continent having avg(LifeExpectancy) < 'terminal'
select SurfaceArea,Name from country order by SurfaceArea desc limit 1
select SurfaceArea,Name from country order by SurfaceArea desc limit 1
select Name from country order by Population desc limit 1
select Name from country order by Population desc limit 1
select Name from country order by Population asc limit 1
select Name from country order by Population asc limit 1
select count(*) from country where Continent = 'terminal'
select count(*) from country where Continent = 'terminal'
select Name from country where Population = 'terminal' or Continent = 'terminal'
select Name from country where Population = 'terminal' or Continent = 'terminal'
select sum(Population),avg(SurfaceArea) from country where SurfaceArea > 'terminal' or Continent = 'terminal'
select sum(Population),avg(SurfaceArea) from country where SurfaceArea > 'terminal' or Continent = 'terminal'
select Name from city where Population between 'terminal' and 'terminal'
select Name from city where Population between 'terminal' and 'terminal'
select Language from countrylanguage group by Language order by count(*) desc limit 1
select Language from countrylanguage group by Language order by count(*) desc limit 1
select max(Percentage),Language,CountryCode from countrylanguage group by CountryCode
select max(Percentage),Language,CountryCode from countrylanguage group by CountryCode
select count(*),max(Percentage) from countrylanguage group by CountryCode where Language = 'terminal'
select count(*),max(Percentage) from countrylanguage group by CountryCode where Language = 'terminal'
select CountryCode,max(Percentage) from countrylanguage group by CountryCode where Language = 'terminal'
select CountryCode,max(Percentage) from countrylanguage group by CountryCode where Language = 'terminal'
select count(*) from conductor
select count(*) from conductor
select Name from conductor order by Age asc 
select Name from conductor order by Age asc 
select Name from conductor where Nationality != 'terminal'
select Name from conductor where Nationality != 'terminal'
select Record_Company from orchestra order by Year_of_Founded desc 
select Record_Company from orchestra order by Year_of_Founded desc 
select avg(Attendance) from show
select avg(Attendance) from show
select max(Share),min(Share) from performance where Type != 'terminal'
select max(Share),min(Share) from performance where Type != 'terminal'
select count(Nationality) from conductor
select count(Nationality) from conductor
select Name from conductor order by Year_of_Work desc 
select Name from conductor order by Year_of_Work desc 
select Name from conductor order by Year_of_Work desc limit 1
select Name from conductor order by Year_of_Work desc limit 1
select T1.Name,T2.Orchestra from conductor as T1 join orchestra as T2 on T1.Conductor_ID = T2.Conductor_ID
select T1.Name,T2.Orchestra from conductor as T1 join orchestra as T2 on T1.Conductor_ID = T2.Conductor_ID
select T1.Name from conductor as T1 join orchestra as T2 on T1.Conductor_ID = T2.Conductor_ID group by T2.Conductor_ID having count(*) > 'terminal'
select T1.Name from conductor as T1 join orchestra as T2 on T1.Conductor_ID = T2.Conductor_ID group by T2.Conductor_ID having count(*) > 'terminal'
select T1.Name from conductor as T1 join orchestra as T2 on T1.Conductor_ID = T2.Conductor_ID group by T2.Conductor_ID order by count(*) desc limit 1
select T1.Name from conductor as T1 join orchestra as T2 on T1.Conductor_ID = T2.Conductor_ID group by T2.Conductor_ID order by count(*) desc limit 1
select T1.Name from conductor as T1 join orchestra as T2 on T1.Conductor_ID = T2.Conductor_ID where T2.Year_of_Founded > 'terminal'
select T1.Name from conductor as T1 join orchestra as T2 on T1.Conductor_ID = T2.Conductor_ID where T2.Year_of_Founded > 'terminal'
select count(*),Record_Company from orchestra group by Record_Company
select count(*),Record_Company from orchestra group by Record_Company
select Major_Record_Format from orchestra group by Major_Record_Format order by count(*) asc 
select Major_Record_Format from orchestra group by Major_Record_Format order by count(*) asc 
select Record_Company from orchestra group by Record_Company order by count(*) desc limit 1
select Record_Company from orchestra group by Record_Company order by count(*) desc limit 1
select Orchestra from orchestra where Orchestra_ID not in (select Orchestra_ID from performance)
select Orchestra from orchestra where Orchestra_ID not in (select Orchestra_ID from performance)
select Record_Company from orchestra where Year_of_Founded < 'terminal' intersect select Record_Company from orchestra where Year_of_Founded > 'terminal'
select Record_Company from orchestra where Year_of_Founded < 'terminal' intersect select Record_Company from orchestra where Year_of_Founded > 'terminal'
select count(*) from orchestra where Major_Record_Format != 'terminal' and Major_Record_Format = 'terminal'
select count(*) from orchestra where Major_Record_Format > 'terminal' and Major_Record_Format = 'terminal'
select Year_of_Founded from orchestra group by Orchestra_ID having count(*) > 'terminal'
select Year_of_Founded from orchestra group by Orchestra_ID having count(*) > 'terminal'
select count(*) from Highschooler
select count(*) from Highschooler
select name,grade from Highschooler
select name,grade from Highschooler
select grade from Highschooler
select grade from Highschooler
select grade from Highschooler where name = 'terminal'
select grade from Highschooler where name = 'terminal'
select name from Highschooler where grade = 'terminal'
select name from Highschooler where grade = 'terminal'
select ID from Highschooler where name = 'terminal'
select ID from Highschooler where name = 'terminal'
select count(*) from Highschooler where grade > 'terminal' and grade = 'terminal'
select count(*) from Highschooler where grade > 'terminal' and grade = 'terminal'
select count(*),grade from Highschooler group by grade
select count(*),grade from Highschooler group by grade
select grade from Highschooler group by grade order by count(*) desc limit 1
select grade from Highschooler group by grade order by count(*) desc limit 1
select grade from Highschooler group by grade having count(*) >= 'terminal'
select grade from Highschooler group by grade having count(*) >= 'terminal'
select student_id,count(*) from Friend group by student_id
select count(*),student_id from Friend group by student_id
select name,count(*) from Highschooler group by ID
select name,count(*) from Highschooler group by ID
select name from Highschooler group by ID order by count(*) desc limit 1
select name from Highschooler group by ID order by count(*) desc limit 1
select name from Highschooler group by ID having count(*) >= 'terminal'
select name from Highschooler group by ID having count(*) >= 'terminal'
select name from Highschooler where name = 'terminal'
select name from Highschooler where name = 'terminal'
select count(*) from Highschooler where name = 'terminal'
select count(*) from Highschooler where name = 'terminal'
select ID from Highschooler except select student_id from Friend
select ID from Highschooler except select student_id from Friend
select name from Highschooler except select name from Highschooler
select name from Highschooler except select name from Highschooler
select student_id from Friend intersect select liked_id from Likes
select student_id from Friend intersect select liked_id from Likes
select name from Highschooler intersect select name from Highschooler
select name from Highschooler intersect select name from Highschooler
select student_id,count(*) from Likes group by student_id
select student_id,count(*) from Likes group by student_id
select count(*),name from Highschooler group by ID
select count(*),name from Highschooler group by ID having count(*) >= 'terminal'
select name from Highschooler group by ID order by count(*) desc limit 1
select name from Highschooler group by ID order by count(*) desc limit 1
select name from Highschooler group by ID having count(*) >= 'terminal'
select name from Highschooler group by ID having count(*) >= 'terminal'
select name from Highschooler group by ID where grade > 'terminal' having count(*) >= 'terminal'
select name from Highschooler group by ID where grade > 'terminal' having count(*) >= 'terminal'
select count(*) from Highschooler where name = 'terminal'
select count(*) from Highschooler where name = 'terminal'
select avg(grade) from Highschooler where ID in (select ID from Highschooler)
select avg(grade) from Highschooler where ID in (select ID from Highschooler)
select min(grade) from Highschooler where ID not in (select ID from Highschooler)
select min(grade) from Highschooler where ID not in (select ID from Highschooler)
select state from Owners intersect select state from Professionals
select state from Owners intersect select state from Professionals
select avg(age) from Dogs where dog_id in (select dog_id from Treatments)
select avg(age) from Dogs where dog_id in (select dog_id from Treatments)
select last_name,cell_number,professional_id from Professionals where state = 'terminal' union select T1.cell_number,T2.professional_id,T1.professional_id from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id group by T1.professional_id having count(*) > 'terminal'
select last_name,cell_number,professional_id from Professionals where state = 'terminal' union select T1.cell_number,T2.professional_id,T1.professional_id from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id group by T1.professional_id having count(*) > 'terminal'
select name from Dogs where dog_id not in (select dog_id from Treatments group by dog_id having sum(cost_of_treatment) > 'terminal')
select name from Dogs where dog_id not in (select dog_id from Treatments group by dog_id having sum(cost_of_treatment) > 'terminal')
select first_name from Professionals union select name from Dogs
select first_name from Professionals union select name from Dogs
select email_address,role_code,professional_id from Professionals except select T2.professional_id,T1.role_code,T1.professional_id from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id
select email_address,role_code,professional_id from Professionals except select T1.role_code,T1.professional_id,T2.professional_id from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id
select owner_id,first_name,last_name from Owners group by owner_id order by count(*) desc limit 1
select owner_id,first_name,last_name from Owners group by owner_id order by count(*) desc limit 1
select T2.professional_id,T1.role_code,T1.first_name from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id group by T2.professional_id having count(*) >= 'terminal'
select T2.professional_id,T1.role_code,T1.first_name from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id group by T2.professional_id having count(*) >= 'terminal'
select breed_name from Breeds group by breed_name order by count(*) desc limit 1
select breed_name from Breeds group by breed_name order by count(*) desc limit 1
select owner_id,last_name from Owners group by owner_id order by count(*) desc limit 1
select last_name,owner_id from Owners group by owner_id order by count(*) desc limit 1
select T1.treatment_type_description from Treatment_Types as T1 join Treatments as T2 on T1.treatment_type_code = T2.treatment_type_code group by T1.treatment_type_code order by sum(T2.cost_of_treatment) asc limit 1
select T1.treatment_type_description from Treatment_Types as T1 join Treatments as T2 on T1.treatment_type_code = T2.treatment_type_code group by T1.treatment_type_code order by sum(T2.cost_of_treatment) asc limit 1
select T1.owner_id,T1.zip_code from Owners as T1 join Dogs as T2 on T1.owner_id = T2.owner_id join Treatments as T3 on T2.dog_id = T3.dog_id group by T1.owner_id order by sum(T3.cost_of_treatment) desc limit 1
select T1.zip_code,T1.owner_id from Owners as T1 join Dogs as T2 on T1.owner_id = T2.owner_id join Treatments as T3 on T2.dog_id = T3.dog_id group by T1.owner_id order by sum(T3.cost_of_treatment) desc limit 1
select T1.professional_id,T2.professional_id from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id group by T2.professional_id having count(*) >= 'terminal'
select T1.professional_id,T2.professional_id from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id group by T2.professional_id having count(*) >= 'terminal'
select T1.first_name,T1.last_name from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id where T2.cost_of_treatment < (select avg(cost_of_treatment) from Treatments)
select T1.last_name,T1.first_name from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id where T2.cost_of_treatment < (select avg(cost_of_treatment) from Treatments)
select T2.date_of_treatment,T1.first_name from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id
select T1.first_name,T2.date_of_treatment from Professionals as T1 join Treatments as T2 on T1.professional_id = T2.professional_id
select T1.treatment_type_description,T2.cost_of_treatment from Treatment_Types as T1 join Treatments as T2 on T1.treatment_type_code = T2.treatment_type_code
select T2.cost_of_treatment,T1.treatment_type_description from Treatment_Types as T1 join Treatments as T2 on T1.treatment_type_code = T2.treatment_type_code
select T2.size_code,T3.last_name,T1.size_code from Sizes as T1 join Dogs as T2 on T1.size_code = T2.size_code join Owners as T3 on T2.owner_id = T3.owner_id
select T3.last_name,T2.size_code,T1.size_code from Sizes as T1 join Dogs as T2 on T1.size_code = T2.size_code join Owners as T3 on T2.owner_id = T3.owner_id
select T1.first_name,T2.name from Owners as T1 join Dogs as T2 on T1.owner_id = T2.owner_id
select T1.first_name,T2.name from Owners as T1 join Dogs as T2 on T1.owner_id = T2.owner_id
select T2.date_of_treatment,T1.name from Dogs as T1 join Treatments as T2 on T1.dog_id = T2.dog_id where T1.breed_code = (select breed_code from Dogs group by breed_code order by count(*) asc limit 1)
select T2.date_of_treatment,T1.name from Dogs as T1 join Treatments as T2 on T1.dog_id = T2.dog_id where T1.breed_code = (select breed_code from Dogs group by breed_code order by count(*) asc limit 1)
select T2.name,T1.first_name from Owners as T1 join Dogs as T2 on T1.owner_id = T2.owner_id where T1.state = 'terminal'
select T1.first_name,T2.name from Owners as T1 join Dogs as T2 on T1.owner_id = T2.owner_id where T1.state = 'terminal'
select date_arrived,date_departed from Dogs
select date_arrived,date_departed from Dogs
select T1.last_name from Owners as T1 join Dogs as T2 on T1.owner_id = T2.owner_id where T2.age = (select max(age) from Dogs)
select T1.last_name from Owners as T1 join Dogs as T2 on T1.owner_id = T2.owner_id where T2.age = (select max(age) from Dogs)
select email_address from Professionals where state like 'terminal' and state = 'terminal'
select email_address from Professionals where state like 'terminal' and state = 'terminal'
select date_arrived,date_departed from Dogs
select date_arrived,date_departed from Dogs
select count(dog_id) from Treatments
select count(dog_id) from Treatments
select count(professional_id) from Treatments
select count(professional_id) from Treatments
select state,street,city,role_code from Professionals where city like 'terminal'
select state,street,role_code,city from Professionals where city like 'terminal'
select email_address,first_name,last_name from Owners where state like 'terminal'
select email_address,first_name,last_name from Owners where state like 'terminal'
select count(*) from Dogs where age < (select avg(age) from Dogs)
select count(*) from Dogs where age < (select avg(age) from Dogs)
select cost_of_treatment from Treatments order by date_of_treatment desc limit 1
select cost_of_treatment from Treatments order by date_of_treatment desc limit 1
select count(*) from Dogs where dog_id not in (select dog_id from Treatments)
select count(*) from Dogs where dog_id not in (select dog_id from Treatments)
select count(*) from Owners where owner_id not in (select owner_id from Dogs)
select count(*) from Owners where owner_id not in (select owner_id from Dogs)
select count(*) from Professionals where professional_id not in (select professional_id from Treatments)
select count(*) from Professionals where professional_id not in (select professional_id from Treatments)
select weight,age,name from Dogs where abandoned_yn = 'terminal'
select weight,age,name from Dogs where abandoned_yn = 'terminal'
select avg(age) from Dogs
select avg(age) from Dogs
select max(age) from Dogs
select max(age) from Dogs
select charge_type,charge_amount from Charges
select charge_amount,charge_type from Charges
select max(charge_amount) from Charges
select max(charge_amount) from Charges
select home_phone,email_address,cell_number from Professionals
select home_phone,email_address,cell_number from Professionals
select breed_code,size_code from Dogs
select breed_code,size_code from Dogs
select T3.first_name,T1.treatment_type_description from Treatment_Types as T1 join Treatments as T2 on T1.treatment_type_code = T2.treatment_type_code join Professionals as T3 on T2.professional_id = T3.professional_id
select T3.first_name,T1.treatment_type_description from Treatment_Types as T1 join Treatments as T2 on T1.treatment_type_code = T2.treatment_type_code join Professionals as T3 on T2.professional_id = T3.professional_id
select count(*) from singer
select count(*) from singer
select Name from singer order by Net_Worth_Millions asc 
select Name from singer order by Net_Worth_Millions asc 
select Birth_Year,Citizenship from singer
select Birth_Year,Citizenship from singer
select Name from singer where Citizenship != 'terminal'
select Name from singer where Citizenship != 'terminal'
select Name from singer where Birth_Year != 'terminal' and Birth_Year = 'terminal'
select Name from singer where Birth_Year != 'terminal' and Birth_Year = 'terminal'
select Name from singer order by Net_Worth_Millions desc limit 1
select Name from singer order by Net_Worth_Millions desc limit 1
select count(*),Citizenship from singer group by Citizenship
select count(*),Citizenship from singer group by Citizenship
select Citizenship from singer group by Citizenship order by count(*) desc limit 1
select Citizenship from singer group by Citizenship order by count(*) desc limit 1
select max(Net_Worth_Millions),Citizenship from singer group by Citizenship
select max(Net_Worth_Millions),Citizenship from singer group by Citizenship
select T2.Title,T1.Name from singer as T1 join song as T2 on T1.Singer_ID = T2.Singer_ID
select T2.Title,T1.Name from singer as T1 join song as T2 on T1.Singer_ID = T2.Singer_ID
select T1.Name from singer as T1 join song as T2 on T1.Singer_ID = T2.Singer_ID where T2.Sales > 'terminal'
select T1.Name from singer as T1 join song as T2 on T1.Singer_ID = T2.Singer_ID where T2.Sales > 'terminal'
select Name from singer group by Name having count(*) > 'terminal'
select Name from singer group by Name having count(*) > 'terminal'
select sum(T2.Sales),T1.Name from singer as T1 join song as T2 on T1.Singer_ID = T2.Singer_ID group by T1.Name
select sum(T2.Sales),T1.Name from singer as T1 join song as T2 on T1.Singer_ID = T2.Singer_ID group by T1.Name
select Name from singer where Singer_ID not in (select Singer_ID from song)
select Name from singer where Singer_ID not in (select Singer_ID from song)
select Citizenship from singer where Birth_Year < 'terminal' intersect select Citizenship from singer where Birth_Year > 'terminal'
select Citizenship from singer where Birth_Year < 'terminal' intersect select Citizenship from singer where Birth_Year > 'terminal'
select count(*) from Ref_Feature_Types
select T1.feature_type_name from Ref_Feature_Types as T1 join Other_Available_Features as T2 on T1.feature_type_code = T2.feature_type_code where T2.feature_name = 'terminal'
select property_type_description from Ref_Property_Types group by property_type_code
select property_name from Properties where property_type_code = 'terminal' union select property_name from Properties where room_count > 'terminal' or property_type_code = 'terminal'
