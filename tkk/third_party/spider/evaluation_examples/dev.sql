Question 1:  How many singers do we have ? ||| concert_singer
SQL:  select count(*) from singer

Question 2:  What is the total number of singers ? ||| concert_singer
SQL:  select count(*) from singer

Question 3:  Show name , country , age for all singers ordered by age from the oldest to the youngest . ||| concert_singer
SQL:  select name ,  country ,  age from singer order by age desc

Question 4:  What are the names , countries , and ages for every singer in descending order of age ? ||| concert_singer
SQL:  select name ,  country ,  age from singer order by age desc

Question 5:  What is the average , minimum , and maximum age of all singers from France ? ||| concert_singer
SQL:  select avg(age) ,  min(age) ,  max(age) from singer where country  =  'france'

Question 6:  What is the average , minimum , and maximum age for all French singers ? ||| concert_singer
SQL:  select avg(age) ,  min(age) ,  max(age) from singer where country  =  'france'

Question 7:  Show the name and the release year of the song by the youngest singer . ||| concert_singer
SQL:  select song_name ,  song_release_year from singer order by age limit 1

Question 8:  What are the names and release years for all the songs of the youngest singer ? ||| concert_singer
SQL:  select song_name ,  song_release_year from singer order by age limit 1

Question 9:  What are all distinct countries where singers above age 20 are from ? ||| concert_singer
SQL:  select distinct country from singer where age  >  20

Question 10:  What are the different countries with singers above age 20 ? ||| concert_singer
SQL:  select distinct country from singer where age  >  20

Question 11:  Show all countries and the number of singers in each country . ||| concert_singer
SQL:  select country ,  count(*) from singer group by country

Question 12:  How many singers are from each country ? ||| concert_singer
SQL:  select country ,  count(*) from singer group by country

Question 13:  List all song names by singers above the average age . ||| concert_singer
SQL:  select song_name from singer where age  >  (select avg(age) from singer)

Question 14:  What are all the song names by singers who are older than average ? ||| concert_singer
SQL:  select song_name from singer where age  >  (select avg(age) from singer)

Question 15:  Show location and name for all stadiums with a capacity between 5000 and 10000 . ||| concert_singer
SQL:  select location ,  name from stadium where capacity between 5000 and 10000

Question 16:  What are the locations and names of all stations with capacity between 5000 and 10000 ? ||| concert_singer
SQL:  select location ,  name from stadium where capacity between 5000 and 10000

Question 17:  What is the maximum capacity and the average of all stadiums ? ||| concert_singer
SQL:  select max(capacity), average from stadium

Question 18:  What is the average and maximum capacities for all stadiums ? ||| concert_singer
SQL:  select avg(capacity) ,  max(capacity) from stadium

Question 19:  What is the name and capacity for the stadium with highest average attendance ? ||| concert_singer
SQL:  select name ,  capacity from stadium order by average desc limit 1

Question 20:  What is the name and capacity for the stadium with the highest average attendance ? ||| concert_singer
SQL:  select name ,  capacity from stadium order by average desc limit 1

Question 21:  How many concerts are there in year 2014 or 2015 ? ||| concert_singer
SQL:  select count(*) from concert where year  =  2014 or year  =  2015

Question 22:  How many concerts occurred in 2014 or 2015 ? ||| concert_singer
SQL:  select count(*) from concert where year  =  2014 or year  =  2015

Question 23:  Show the stadium name and the number of concerts in each stadium . ||| concert_singer
SQL:  select t2.name ,  count(*) from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id group by t1.stadium_id

Question 24:  For each stadium , how many concerts play there ? ||| concert_singer
SQL:  select t2.name ,  count(*) from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id group by t1.stadium_id

Question 25:  Show the stadium name and capacity with most number of concerts in year 2014 or after . ||| concert_singer
SQL:  select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >=  2014 group by t2.stadium_id order by count(*) desc limit 1

Question 26:  What is the name and capacity of the stadium with the most concerts after 2013 ? ||| concert_singer
SQL:  select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >  2013 group by t2.stadium_id order by count(*) desc limit 1

Question 27:  Which year has most number of concerts ? ||| concert_singer
SQL:  select year from concert group by year order by count(*) desc limit 1

Question 28:  What is the year that had the most concerts ? ||| concert_singer
SQL:  select year from concert group by year order by count(*) desc limit 1

Question 29:  Show the stadium names without any concert . ||| concert_singer
SQL:  select name from stadium where stadium_id not in (select stadium_id from concert)

Question 30:  What are the names of the stadiums without any concerts ? ||| concert_singer
SQL:  select name from stadium where stadium_id not in (select stadium_id from concert)

Question 31:  Show countries where a singer above age 40 and a singer below 30 are from . ||| concert_singer
SQL:  select country from singer where age  >  40 intersect select country from singer where age  <  30

Question 32:  Show names for all stadiums except for stadiums having a concert in year 2014 . ||| concert_singer
SQL:  select name from stadium except select t2.name from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014

Question 33:  What are the names of all stadiums that did not have a concert in 2014 ? ||| concert_singer
SQL:  select name from stadium except select t2.name from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014

Question 34:  Show the name and theme for all concerts and the number of singers in each concert . ||| concert_singer
SQL:  select t2.concert_name ,  t2.theme ,  count(*) from singer_in_concert as t1 join concert as t2 on t1.concert_id  =  t2.concert_id group by t2.concert_id

Question 35:  What are the names , themes , and number of singers for every concert ? ||| concert_singer
SQL:  select t2.concert_name ,  t2.theme ,  count(*) from singer_in_concert as t1 join concert as t2 on t1.concert_id  =  t2.concert_id group by t2.concert_id

Question 36:  List singer names and number of concerts for each singer . ||| concert_singer
SQL:  select t2.name ,  count(*) from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id group by t2.singer_id

Question 37:  What are the names of the singers and number of concerts for each person ? ||| concert_singer
SQL:  select t2.name ,  count(*) from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id group by t2.singer_id

Question 38:  List all singer names in concerts in year 2014 . ||| concert_singer
SQL:  select t2.name from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id join concert as t3 on t1.concert_id  =  t3.concert_id where t3.year  =  2014

Question 39:  What are the names of the singers who performed in a concert in 2014 ? ||| concert_singer
SQL:  select t2.name from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id join concert as t3 on t1.concert_id  =  t3.concert_id where t3.year  =  2014

Question 40:  what is the name and nation of the singer who have a song having 'Hey ' in its name ? ||| concert_singer
SQL:  select name ,  country from singer where song_name like '%hey%'

Question 41:  What is the name and country of origin of every singer who has a song with the word 'Hey ' in its title ? ||| concert_singer
SQL:  select name ,  country from singer where song_name like '%hey%'

Question 42:  Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015 . ||| concert_singer
SQL:  select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014 intersect select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2015

Question 43:  What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015 ? ||| concert_singer
SQL:  select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014 intersect select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2015

Question 44:  Find the number of concerts happened in the stadium with the highest capacity . ||| concert_singer
SQL:  select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)

Question 45:  What are the number of concerts that occurred in the stadium with the largest capacity ? ||| concert_singer
SQL:  select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)

Question 46:  Find the number of pets whose weight is heavier than 10 . ||| pets_1
SQL:  select count(*) from pets where weight  >  10

Question 47:  How many pets have a greater weight than 10 ? ||| pets_1
SQL:  select count(*) from pets where weight  >  10

Question 48:  Find the weight of the youngest dog . ||| pets_1
SQL:  select weight from pets order by pet_age limit 1

Question 49:  How much does the youngest dog weigh ? ||| pets_1
SQL:  select weight from pets order by pet_age limit 1

Question 50:  Find the maximum weight for each type of pet . List the maximum weight and pet type . ||| pets_1
SQL:  select max(weight) ,  pettype from pets group by pettype

Question 51:  List the maximum weight and type for each type of pet . ||| pets_1
SQL:  select max(weight) ,  pettype from pets group by pettype

Question 52:  Find number of pets owned by students who are older than 20 . ||| pets_1
SQL:  select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.age  >  20

Question 53:  How many pets are owned by students that have an age greater than 20 ? ||| pets_1
SQL:  select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.age  >  20

Question 54:  Find the number of dog pets that are raised by female students ( with sex F ) . ||| pets_1
SQL:  select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t2.petid  =  t3.petid where t1.sex  =  'f' and t3.pettype  =  'dog'

Question 55:  How many dog pets are raised by female students ? ||| pets_1
SQL:  select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t2.petid  =  t3.petid where t1.sex  =  'f' and t3.pettype  =  'dog'

Question 56:  Find the number of distinct type of pets . ||| pets_1
SQL:  select count(distinct pettype) from pets

Question 57:  How many different types of pet are there ? ||| pets_1
SQL:  select count(distinct pettype) from pets

Question 58:  Find the first name of students who have cat or dog pet . ||| pets_1
SQL:  select distinct t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' or t3.pettype  =  'dog'

Question 59:  What are the first names of every student who has a cat or dog as a pet ? ||| pets_1
SQL:  select distinct t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' or t3.pettype  =  'dog'

Question 60:  Find the first name of students who have both cat and dog pets . ||| pets_1
SQL:  select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' intersect select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog'

Question 61:  What are the students ' first names who have both cats and dogs as pets ? ||| pets_1
SQL:  select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' intersect select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog'

Question 62:  Find the major and age of students who do not have a cat pet . ||| pets_1
SQL:  select major ,  age from student where stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')

Question 63:  What major is every student who does not own a cat as a pet , and also how old are they ? ||| pets_1
SQL:  select major ,  age from student where stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')

Question 64:  Find the id of students who do not have a cat pet . ||| pets_1
SQL:  select stuid from student except select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat'

Question 65:  What are the ids of the students who do not own cats as pets ? ||| pets_1
SQL:  select stuid from student except select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat'

Question 66:  Find the first name and age of students who have a dog but do not have a cat as a pet . ||| pets_1
SQL:  select t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog' and t1.stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')

Question 67:  What is the first name of every student who has a dog but does not have a cat ? ||| pets_1
SQL:  select t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog' and t1.stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')

Question 68:  Find the type and weight of the youngest pet . ||| pets_1
SQL:  select pettype ,  weight from pets order by pet_age limit 1

Question 69:  What type of pet is the youngest animal , and how much does it weigh ? ||| pets_1
SQL:  select pettype ,  weight from pets order by pet_age limit 1

Question 70:  Find the id and weight of all pets whose age is older than 1 . ||| pets_1
SQL:  select petid ,  weight from pets where pet_age  >  1

Question 71:  What is the id and weight of every pet who is older than 1 ? ||| pets_1
SQL:  select petid ,  weight from pets where pet_age  >  1

Question 72:  Find the average and maximum age for each type of pet . ||| pets_1
SQL:  select avg(pet_age) ,  max(pet_age) ,  pettype from pets group by pettype

Question 73:  What is the average and maximum age for each pet type ? ||| pets_1
SQL:  select avg(pet_age) ,  max(pet_age) ,  pettype from pets group by pettype

Question 74:  Find the average weight for each pet type . ||| pets_1
SQL:  select avg(weight) ,  pettype from pets group by pettype

Question 75:  What is the average weight for each type of pet ? ||| pets_1
SQL:  select avg(weight) ,  pettype from pets group by pettype

Question 76:  Find the first name and age of students who have a pet . ||| pets_1
SQL:  select distinct t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid

Question 77:  What are the different first names and ages of the students who do have pets ? ||| pets_1
SQL:  select distinct t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid

Question 78:  Find the id of the pet owned by student whose last name is ‘Smith’ . ||| pets_1
SQL:  select t2.petid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.lname  =  'smith'

Question 79:  What is the id of the pet owned by the student whose last name is 'Smith ' ? ||| pets_1
SQL:  select t2.petid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.lname  =  'smith'

Question 80:  Find the number of pets for each student who has any pet and student id . ||| pets_1
SQL:  select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid

Question 81:  For students who have pets , how many pets does each student have ? list their ids instead of names . ||| pets_1
SQL:  select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid

Question 82:  Find the first name and gender of student who have more than one pet . ||| pets_1
SQL:  select t1.fname ,  t1.sex from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid having count(*)  >  1

Question 83:  What is the first name and gender of the all the students who have more than one pet ? ||| pets_1
SQL:  select t1.fname ,  t1.sex from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid having count(*)  >  1

Question 84:  Find the last name of the student who has a cat that is age 3 . ||| pets_1
SQL:  select t1.lname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pet_age  =  3 and t3.pettype  =  'cat'

Question 85:  What is the last name of the student who has a cat that is 3 years old ? ||| pets_1
SQL:  select t1.lname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pet_age  =  3 and t3.pettype  =  'cat'

Question 86:  Find the average age of students who do not have any pet . ||| pets_1
SQL:  select avg(age) from student where stuid not in (select stuid from has_pet)

Question 87:  What is the average age for all students who do not own any pets ? ||| pets_1
SQL:  select avg(age) from student where stuid not in (select stuid from has_pet)

Question 88:  How many continents are there ? ||| car_1
SQL:  select count(*) from continents;

Question 89:  What is the number of continents ? ||| car_1
SQL:  select count(*) from continents;

Question 90:  How many countries does each continent have ? List the continent id , continent name and the number of countries . ||| car_1
SQL:  select t1.contid ,  t1.continent ,  count(*) from continents as t1 join countries as t2 on t1.contid  =  t2.continent group by t1.contid;

Question 91:  For each continent , list its id , name , and how many countries it has ? ||| car_1
SQL:  select t1.contid ,  t1.continent ,  count(*) from continents as t1 join countries as t2 on t1.contid  =  t2.continent group by t1.contid;

Question 92:  How many countries are listed ? ||| car_1
SQL:  select count(*) from countries;

Question 93:  How many countries exist ? ||| car_1
SQL:  select count(*) from countries;

Question 94:  How many models does each car maker produce ? List maker full name , id and the number . ||| car_1
SQL:  select t1.fullname ,  t1.id ,  count(*) from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id;

Question 95:  What is the full name of each car maker , along with its id and how many models it produces ? ||| car_1
SQL:  select t1.fullname ,  t1.id ,  count(*) from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id;

Question 96:  Which model of the car has the minimum horsepower ? ||| car_1
SQL:  select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.horsepower asc limit 1;

Question 97:  What is the model of the car with the smallest amount of horsepower ? ||| car_1
SQL:  select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.horsepower asc limit 1;

Question 98:  Find the model of the car whose weight is below the average weight . ||| car_1
SQL:  select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id where t2.weight  <  (select avg(weight) from cars_data)

Question 99:  What is the model for the car with a weight smaller than the average ? ||| car_1
SQL:  select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id where t2.weight  <  (select avg(weight) from cars_data)

Question 100:  Find the name of the makers that produced some cars in the year of 1970 ? ||| car_1
SQL:  select distinct t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker join car_names as t3 on t2.model  =  t3.model join cars_data as t4 on t3.makeid  =  t4.id where t4.year  =  '1970';

Question 101:  What is the name of the different car makers who produced a car in 1970 ? ||| car_1
SQL:  select distinct t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker join car_names as t3 on t2.model  =  t3.model join cars_data as t4 on t3.makeid  =  t4.id where t4.year  =  '1970';

Question 102:  Find the make and production time of the cars that were produced in the earliest year ? ||| car_1
SQL:  select t2.make ,  t1.year from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t1.year  =  (select min(year) from cars_data);

Question 103:  What is the maker of the carr produced in the earliest year and what year was it ? ||| car_1
SQL:  select t2.make ,  t1.year from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t1.year  =  (select min(year) from cars_data);

Question 104:  Which distinct car models are the produced after 1980 ? ||| car_1
SQL:  select distinct t1.model from model_list as t1 join car_names as t2 on t1.model  =  t2.model join cars_data as t3 on t2.makeid  =  t3.id where t3.year  >  1980;

Question 105:  What are the different models for the cards produced after 1980 ? ||| car_1
SQL:  select distinct t1.model from model_list as t1 join car_names as t2 on t1.model  =  t2.model join cars_data as t3 on t2.makeid  =  t3.id where t3.year  >  1980;

Question 106:  How many car makers are there in each continents ? List the continent name and the count . ||| car_1
SQL:  select t1.continent ,  count(*) from continents as t1 join countries as t2 on t1.contid  =  t2.continent join car_makers as t3 on t2.countryid  =  t3.country group by t1.continent;

Question 107:  What is the name of each continent and how many car makers are there in each one ? ||| car_1
SQL:  select t1.continent ,  count(*) from continents as t1 join countries as t2 on t1.contid  =  t2.continent join car_makers as t3 on t2.countryid  =  t3.country group by t1.continent;

Question 108:  Which of the countries has the most car makers ? List the country name . ||| car_1
SQL:  select t2.countryname from car_makers as t1 join countries as t2 on t1.country  =  t2.countryid group by t1.country order by count(*) desc limit 1;

Question 109:  What is the name of the country with the most car makers ? ||| car_1
SQL:  select t2.countryname from car_makers as t1 join countries as t2 on t1.country  =  t2.countryid group by t1.country order by count(*) desc limit 1;

Question 110:  How many car models are produced by each maker ? Only list the count and the maker full name . ||| car_1
SQL:  select count(*) ,  t2.fullname from model_list as t1 join car_makers as t2 on t1.maker  =  t2.id group by t2.id;

Question 111:  What is the number of car models that are produced by each maker and what is the id and full name of each maker ? ||| car_1
SQL:  select count(*) ,  t2.fullname ,  t2.id from model_list as t1 join car_makers as t2 on t1.maker  =  t2.id group by t2.id;

Question 112:  What is the accelerate of the car make amc hornet sportabout ( sw ) ? ||| car_1
SQL:  select t1.accelerate from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t2.make  =  'amc hornet sportabout (sw)';

Question 113:  How much does the car accelerate that makes amc hornet sportabout ( sw ) ? ||| car_1
SQL:  select t1.accelerate from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t2.make  =  'amc hornet sportabout (sw)';

Question 114:  How many car makers are there in france ? ||| car_1
SQL:  select count(*) from car_makers as t1 join countries as t2 on t1.country  =  t2.countryid where t2.countryname  =  'france';

Question 115:  What is the number of makers of care in France ? ||| car_1
SQL:  select count(*) from car_makers as t1 join countries as t2 on t1.country  =  t2.countryid where t2.countryname  =  'france';

Question 116:  How many car models are produced in the usa ? ||| car_1
SQL:  select count(*) from model_list as t1 join car_makers as t2 on t1.maker  =  t2.id join countries as t3 on t2.country  =  t3.countryid where t3.countryname  =  'usa';

Question 117:  What is the count of the car models produced in the United States ? ||| car_1
SQL:  select count(*) from model_list as t1 join car_makers as t2 on t1.maker  =  t2.id join countries as t3 on t2.country  =  t3.countryid where t3.countryname  =  'usa';

Question 118:  What is the average miles per gallon ( mpg ) of the cars with 4 cylinders ? ||| car_1
SQL:  select avg(mpg) from cars_data where cylinders  =  4;

Question 119:  What is the average miles per gallon of all the cards with 4 cylinders ? ||| car_1
SQL:  select avg(mpg) from cars_data where cylinders  =  4;

Question 120:  What is the smallest weight of the car produced with 8 cylinders on 1974 ? ||| car_1
SQL:  select min(weight) from cars_data where cylinders  =  8 and year  =  1974

Question 121:  What is the minimum weight of the car with 8 cylinders produced in 1974 ? ||| car_1
SQL:  select min(weight) from cars_data where cylinders  =  8 and year  =  1974

Question 122:  What are all the makers and models ? ||| car_1
SQL:  select maker ,  model from model_list;

Question 123:  What are the makers and models ? ||| car_1
SQL:  select maker ,  model from model_list;

Question 124:  What are the countries having at least one car maker ? List name and id . ||| car_1
SQL:  select t1.countryname ,  t1.countryid from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >=  1;

Question 125:  What are the names and ids of all countries with at least one car maker ? ||| car_1
SQL:  select t1.countryname ,  t1.countryid from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >=  1;

Question 126:  What is the number of the cars with horsepower more than 150 ? ||| car_1
SQL:  select count(*) from cars_data where horsepower  >  150;

Question 127:  What is the number of cars with a horsepower greater than 150 ? ||| car_1
SQL:  select count(*) from cars_data where horsepower  >  150;

Question 128:  What is the average weight of cars each year ? ||| car_1
SQL:  select avg(weight) ,  year from cars_data group by year;

Question 129:  What is the average weight and year for each year ? ||| car_1
SQL:  select avg(weight) ,  year from cars_data group by year;

Question 130:  Which countries in europe have at least 3 car manufacturers ? ||| car_1
SQL:  select t1.countryname from countries as t1 join continents as t2 on t1.continent  =  t2.contid join car_makers as t3 on t1.countryid  =  t3.country where t2.continent  =  'europe' group by t1.countryname having count(*)  >=  3;

Question 131:  What are the names of all European countries with at least 3 manufacturers ? ||| car_1
SQL:  select t1.countryname from countries as t1 join continents as t2 on t1.continent  =  t2.contid join car_makers as t3 on t1.countryid  =  t3.country where t2.continent  =  'europe' group by t1.countryname having count(*)  >=  3;

Question 132:  What is the maximum horsepower and the make of the car models with 3 cylinders ? ||| car_1
SQL:  select t2.horsepower ,  t1.make from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id where t2.cylinders  =  3 order by t2.horsepower desc limit 1;

Question 133:  What is the largest amount of horsepower for the models with 3 cylinders and what make is it ? ||| car_1
SQL:  select t2.horsepower ,  t1.make from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id where t2.cylinders  =  3 order by t2.horsepower desc limit 1;

Question 134:  Which model saves the most gasoline ? That is to say , have the maximum miles per gallon . ||| car_1
SQL:  select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.mpg desc limit 1;

Question 135:  What is the car model with the highest mpg ? ||| car_1
SQL:  select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.mpg desc limit 1;

Question 136:  What is the average horsepower of the cars before 1980 ? ||| car_1
SQL:  select avg(horsepower) from cars_data where year  <  1980;

Question 137:  What is the average horsepower for all cars produced before 1980 ? ||| car_1
SQL:  select avg(horsepower) from cars_data where year  <  1980;

Question 138:  What is the average edispl of the cars of model volvo ? ||| car_1
SQL:  select avg(t2.edispl) from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id where t1.model  =  'volvo';

Question 139:  What is the average edispl for all volvos ? ||| car_1
SQL:  select avg(t2.edispl) from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id where t1.model  =  'volvo';

Question 140:  What is the maximum accelerate for different number of cylinders ? ||| car_1
SQL:  select max(accelerate) ,  cylinders from cars_data group by cylinders;

Question 141:  What is the maximum accelerate for all the different cylinders ? ||| car_1
SQL:  select max(accelerate) ,  cylinders from cars_data group by cylinders;

Question 142:  Which model has the most version ( make ) of cars ? ||| car_1
SQL:  select model from car_names group by model order by count(*) desc limit 1;

Question 143:  What model has the most different versions ? ||| car_1
SQL:  select model from car_names group by model order by count(*) desc limit 1;

Question 144:  How many cars have more than 4 cylinders ? ||| car_1
SQL:  select count(*) from cars_data where cylinders  >  4;

Question 145:  What is the number of cars with more than 4 cylinders ? ||| car_1
SQL:  select count(*) from cars_data where cylinders  >  4;

Question 146:  how many cars were produced in 1980 ? ||| car_1
SQL:  select count(*) from cars_data where year  =  1980;

Question 147:  In 1980 , how many cars were made ? ||| car_1
SQL:  select count(*) from cars_data where year  =  1980;

Question 148:  How many car models were produced by the maker with full name American Motor Company ? ||| car_1
SQL:  select count(*) from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker where t1.fullname  =  'american motor company';

Question 149:  What is the number of car models created by the car maker American Motor Company ? ||| car_1
SQL:  select count(*) from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker where t1.fullname  =  'american motor company';

Question 150:  Which makers designed more than 3 car models ? List full name and the id . ||| car_1
SQL:  select t1.fullname ,  t1.id from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id having count(*)  >  3;

Question 151:  What are the names and ids of all makers with more than 3 models ? ||| car_1
SQL:  select t1.fullname ,  t1.id from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id having count(*)  >  3;

Question 152:  Which distinctive models are produced by maker with the full name General Motors or weighing more than 3500 ? ||| car_1
SQL:  select distinct t2.model from car_names as t1 join model_list as t2 on t1.model  =  t2.model join car_makers as t3 on t2.maker  =  t3.id join cars_data as t4 on t1.makeid  =  t4.id where t3.fullname  =  'general motors' or t4.weight  >  3500;

Question 153:  What are the different models created by either the car maker General Motors or weighed more than 3500 ? ||| car_1
SQL:  select distinct t2.model from car_names as t1 join model_list as t2 on t1.model  =  t2.model join car_makers as t3 on t2.maker  =  t3.id join cars_data as t4 on t1.makeid  =  t4.id where t3.fullname  =  'general motors' or t4.weight  >  3500;

Question 154:  In which years cars were produced weighing no less than 3000 and no more than 4000 ? ||| car_1
SQL:  select distinct year from cars_data where weight between 3000 and 4000;

Question 155:  What are the different years in which there were cars produced that weighed less than 4000 and also cars that weighted more than 3000 ? ||| car_1
SQL:  select distinct year from cars_data where weight between 3000 and 4000;

Question 156:  What is the horsepower of the car with the largest accelerate ? ||| car_1
SQL:  select t1.horsepower from cars_data as t1 order by t1.accelerate desc limit 1;

Question 157:  What is the horsepower of the car with the greatest accelerate ? ||| car_1
SQL:  select t1.horsepower from cars_data as t1 order by t1.accelerate desc limit 1;

Question 158:  For model volvo , how many cylinders does the car with the least accelerate have ? ||| car_1
SQL:  select t1.cylinders from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t2.model  =  'volvo' order by t1.accelerate asc limit 1;

Question 159:  For a volvo model , how many cylinders does the version with least accelerate have ? ||| car_1
SQL:  select t1.cylinders from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t2.model  =  'volvo' order by t1.accelerate asc limit 1;

Question 160:  How many cars have a larger accelerate than the car with the largest horsepower ? ||| car_1
SQL:  select count(*) from cars_data where accelerate  >  ( select accelerate from cars_data order by horsepower desc limit 1 );

Question 161:  What is the number of cars with a greater accelerate than the one with the most horsepower ? ||| car_1
SQL:  select count(*) from cars_data where accelerate  >  ( select accelerate from cars_data order by horsepower desc limit 1 );

Question 162:  How many countries has more than 2 car makers ? ||| car_1
SQL:  select count(*) from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  2

Question 163:  What is the number of countries with more than 2 car makers ? ||| car_1
SQL:  select count(*) from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  2

Question 164:  How many cars has over 6 cylinders ? ||| car_1
SQL:  select count(*) from cars_data where cylinders  >  6;

Question 165:  What is the number of carsw ith over 6 cylinders ? ||| car_1
SQL:  select count(*) from cars_data where cylinders  >  6;

Question 166:  For the cars with 4 cylinders , which model has the largest horsepower ? ||| car_1
SQL:  select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id where t2.cylinders  =  4 order by t2.horsepower desc limit 1;

Question 167:  For all of the 4 cylinder cars , which model has the most horsepower ? ||| car_1
SQL:  select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id where t2.cylinders  =  4 order by t2.horsepower desc limit 1;

Question 168:  Among the cars with more than lowest horsepower , which ones do not have more than 3 cylinders ? List the car makeid and make name . ||| car_1
SQL:  select t2.makeid ,  t2.make from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t1.horsepower  >  (select min(horsepower) from cars_data) and t1.cylinders  <=  3;

Question 169:  Among the cars that do not have the minimum horsepower , what are the make ids and names of all those with less than 4 cylinders ? ||| car_1
SQL:  select t2.makeid ,  t2.make from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t1.horsepower  >  (select min(horsepower) from cars_data) and t1.cylinders  <  4;

Question 170:  What is the maximum miles per gallon of the car with 8 cylinders or produced before 1980 ? ||| car_1
SQL:  select max(mpg) from cars_data where cylinders  =  8 or year  <  1980

Question 171:  What is the maximum mpg of the cars that had 8 cylinders or that were produced before 1980 ? ||| car_1
SQL:  select max(mpg) from cars_data where cylinders  =  8 or year  <  1980

Question 172:  Which models are lighter than 3500 but not built by the 'Ford Motor Company ' ? ||| car_1
SQL:  select distinct t1.model from model_list as t1 join car_names as t2 on t1.model  =  t2.model join cars_data as t3 on t2.makeid  =  t3.id join car_makers as t4 on t1.maker  =  t4.id where t3.weight  <  3500 and t4.fullname != 'ford motor company';

Question 173:  What are the different models wthat are lighter than 3500 but were not built by the Ford Motor Company ? ||| car_1
SQL:  select distinct t1.model from model_list as t1 join car_names as t2 on t1.model  =  t2.model join cars_data as t3 on t2.makeid  =  t3.id join car_makers as t4 on t1.maker  =  t4.id where t3.weight  <  3500 and t4.fullname != 'ford motor company';

Question 174:  What are the name of the countries where there is not a single car maker ? ||| car_1
SQL:  select countryname from countries except select t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country;

Question 175:  What are the names of the countries with no car makers ? ||| car_1
SQL:  select countryname from countries except select t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country;

Question 176:  Which are the car makers which produce at least 2 models and more than 3 car makers ? List the id and the maker . ||| car_1
SQL:  select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id having count(*)  >=  2 intersect select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker join car_names as t3 on t2.model  =  t3.model group by t1.id having count(*)  >  3;

Question 177:  What are the ids and makers of all car makers that produce at least 2 models and make more than 3 cars ? ||| car_1
SQL:  select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id having count(*)  >=  2 intersect select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker join car_names as t3 on t2.model  =  t3.model group by t1.id having count(*)  >  3;

Question 178:  What are the id and names of the countries which have more than 3 car makers or produce the 'fiat ' model ? ||| car_1
SQL:  select t1.countryid ,  t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  3 union select t1.countryid ,  t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country join model_list as t3 on t2.id  =  t3.maker where t3.model  =  'fiat';

Question 179:  What are the ids and names of all countries that either have more than 3 car makers or produce fiat model ? ||| car_1
SQL:  select t1.countryid ,  t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  3 union select t1.countryid ,  t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country join model_list as t3 on t2.id  =  t3.maker where t3.model  =  'fiat';

Question 180:  Which country does Airline `` JetBlue Airways '' belong to ? ||| flight_2
SQL:  select country from airlines where airline  =  "jetblue airways"

Question 181:  What country is Jetblue Airways affiliated with ? ||| flight_2
SQL:  select country from airlines where airline  =  "jetblue airways"

Question 182:  What is the abbreviation of Airline `` JetBlue Airways '' ? ||| flight_2
SQL:  select abbreviation from airlines where airline  =  "jetblue airways"

Question 183:  Which abbreviation corresponds to Jetblue Airways ? ||| flight_2
SQL:  select abbreviation from airlines where airline  =  "jetblue airways"

Question 184:  List all airline names and their abbreviations in `` USA '' . ||| flight_2
SQL:  select airline ,  abbreviation from airlines where country  =  "usa"

Question 185:  What are the airline names and abbreviations for airlines in the USA ? ||| flight_2
SQL:  select airline ,  abbreviation from airlines where country  =  "usa"

Question 186:  List the airport code and name in the city of Anthony . ||| flight_2
SQL:  select airportcode ,  airportname from airports where city  =  "anthony"

Question 187:  Give the airport code and airport name corresonding to the city Anthony . ||| flight_2
SQL:  select airportcode ,  airportname from airports where city  =  "anthony"

Question 188:  How many airlines do we have ? ||| flight_2
SQL:  select count(*) from airlines

Question 189:  What is the total number of airlines ? ||| flight_2
SQL:  select count(*) from airlines

Question 190:  How many airports do we have ? ||| flight_2
SQL:  select count(*) from airports

Question 191:  Return the number of airports . ||| flight_2
SQL:  select count(*) from airports

Question 192:  How many flights do we have ? ||| flight_2
SQL:  select count(*) from flights

Question 193:  Return the number of flights . ||| flight_2
SQL:  select count(*) from flights

Question 194:  Which airline has abbreviation 'UAL ' ? ||| flight_2
SQL:  select airline from airlines where abbreviation  =  "ual"

Question 195:  Give the airline with abbreviation 'UAL ' . ||| flight_2
SQL:  select airline from airlines where abbreviation  =  "ual"

Question 196:  How many airlines are from USA ? ||| flight_2
SQL:  select count(*) from airlines where country  =  "usa"

Question 197:  Return the number of airlines in the USA . ||| flight_2
SQL:  select count(*) from airlines where country  =  "usa"

Question 198:  Which city and country is the Alton airport at ? ||| flight_2
SQL:  select city ,  country from airports where airportname  =  "alton"

Question 199:  Give the city and country for the Alton airport . ||| flight_2
SQL:  select city ,  country from airports where airportname  =  "alton"

Question 200:  What is the airport name for airport 'AKO ' ? ||| flight_2
SQL:  select airportname from airports where airportcode  =  "ako"

Question 201:  Return the name of the airport with code 'AKO ' . ||| flight_2
SQL:  select airportname from airports where airportcode  =  "ako"

Question 202:  What are airport names at City 'Aberdeen ' ? ||| flight_2
SQL:  select airportname from airports where city = "aberdeen"

Question 203:  What are the names of airports in Aberdeen ? ||| flight_2
SQL:  select airportname from airports where city = "aberdeen"

Question 204:  How many flights depart from 'APG ' ? ||| flight_2
SQL:  select count(*) from flights where sourceairport  =  "apg"

Question 205:  Count the number of flights departing from 'APG ' . ||| flight_2
SQL:  select count(*) from flights where sourceairport  =  "apg"

Question 206:  How many flights have destination ATO ? ||| flight_2
SQL:  select count(*) from flights where destairport  =  "ato"

Question 207:  Count the number of flights into ATO . ||| flight_2
SQL:  select count(*) from flights where destairport  =  "ato"

Question 208:  How many flights depart from City Aberdeen ? ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.sourceairport  =  t2.airportcode where t2.city  =  "aberdeen"

Question 209:  Return the number of flights departing from Aberdeen . ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.sourceairport  =  t2.airportcode where t2.city  =  "aberdeen"

Question 210:  How many flights arriving in Aberdeen city ? ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.destairport  =  t2.airportcode where t2.city  =  "aberdeen"

Question 211:  Return the number of flights arriving in Aberdeen . ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.destairport  =  t2.airportcode where t2.city  =  "aberdeen"

Question 212:  How many flights depart from City 'Aberdeen ' and have destination City 'Ashley ' ? ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.destairport  =  t2.airportcode join airports as t3 on t1.sourceairport  =  t3.airportcode where t2.city  =  "ashley" and t3.city  =  "aberdeen"

Question 213:  How many flights fly from Aberdeen to Ashley ? ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.destairport  =  t2.airportcode join airports as t3 on t1.sourceairport  =  t3.airportcode where t2.city  =  "ashley" and t3.city  =  "aberdeen"

Question 214:  How many flights does airline 'JetBlue Airways ' have ? ||| flight_2
SQL:  select count(*) from flights as t1 join airlines as t2 on t1.airline  =  t2.uid where t2.airline = "jetblue airways"

Question 215:  Give the number of Jetblue Airways flights . ||| flight_2
SQL:  select count(*) from flights as t1 join airlines as t2 on t1.airline  =  t2.uid where t2.airline = "jetblue airways"

Question 216:  How many 'United Airlines ' flights go to Airport 'ASY ' ? ||| flight_2
SQL:  select count(*) from airlines as t1 join flights as t2 on t2.airline  =  t1.uid where t1.airline  =  "united airlines" and t2.destairport  =  "asy"

Question 217:  Count the number of United Airlines flights arriving in ASY Airport . ||| flight_2
SQL:  select count(*) from airlines as t1 join flights as t2 on t2.airline  =  t1.uid where t1.airline  =  "united airlines" and t2.destairport  =  "asy"

Question 218:  How many 'United Airlines ' flights depart from Airport 'AHD ' ? ||| flight_2
SQL:  select count(*) from airlines as t1 join flights as t2 on t2.airline  =  t1.uid where t1.airline  =  "united airlines" and t2.sourceairport  =  "ahd"

Question 219:  Return the number of United Airlines flights leaving from AHD Airport . ||| flight_2
SQL:  select count(*) from airlines as t1 join flights as t2 on t2.airline  =  t1.uid where t1.airline  =  "united airlines" and t2.sourceairport  =  "ahd"

Question 220:  How many United Airlines flights go to City 'Aberdeen ' ? ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.destairport  =  t2.airportcode join airlines as t3 on t3.uid  =  t1.airline where t2.city  =  "aberdeen" and t3.airline  =  "united airlines"

Question 221:  Count the number of United Airlines flights that arrive in Aberdeen . ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.destairport  =  t2.airportcode join airlines as t3 on t3.uid  =  t1.airline where t2.city  =  "aberdeen" and t3.airline  =  "united airlines"

Question 222:  Which city has most number of arriving flights ? ||| flight_2
SQL:  select t1.city from airports as t1 join flights as t2 on t1.airportcode  =  t2.destairport group by t1.city order by count(*) desc limit 1

Question 223:  Which city has the most frequent destination airport ? ||| flight_2
SQL:  select t1.city from airports as t1 join flights as t2 on t1.airportcode  =  t2.destairport group by t1.city order by count(*) desc limit 1

Question 224:  Which city has most number of departing flights ? ||| flight_2
SQL:  select t1.city from airports as t1 join flights as t2 on t1.airportcode  =  t2.sourceairport group by t1.city order by count(*) desc limit 1

Question 225:  Which city is the most frequent source airport ? ||| flight_2
SQL:  select t1.city from airports as t1 join flights as t2 on t1.airportcode  =  t2.sourceairport group by t1.city order by count(*) desc limit 1

Question 226:  What is the code of airport that has the highest number of flights ? ||| flight_2
SQL:  select t1.airportcode from airports as t1 join flights as t2 on t1.airportcode  =  t2.destairport or t1.airportcode  =  t2.sourceairport group by t1.airportcode order by count(*) desc limit 1

Question 227:  What is the airport code of the airport with the most flights ? ||| flight_2
SQL:  select t1.airportcode from airports as t1 join flights as t2 on t1.airportcode  =  t2.destairport or t1.airportcode  =  t2.sourceairport group by t1.airportcode order by count(*) desc limit 1

Question 228:  What is the code of airport that has fewest number of flights ? ||| flight_2
SQL:  select t1.airportcode from airports as t1 join flights as t2 on t1.airportcode  =  t2.destairport or t1.airportcode  =  t2.sourceairport group by t1.airportcode order by count(*) limit 1

Question 229:  Give the code of the airport with the least flights . ||| flight_2
SQL:  select t1.airportcode from airports as t1 join flights as t2 on t1.airportcode  =  t2.destairport or t1.airportcode  =  t2.sourceairport group by t1.airportcode order by count(*) limit 1

Question 230:  Which airline has most number of flights ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline group by t1.airline order by count(*) desc limit 1

Question 231:  What airline serves the most flights ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline group by t1.airline order by count(*) desc limit 1

Question 232:  Find the abbreviation and country of the airline that has fewest number of flights ? ||| flight_2
SQL:  select t1.abbreviation ,  t1.country from airlines as t1 join flights as t2 on t1.uid  =  t2.airline group by t1.airline order by count(*) limit 1

Question 233:  What is the abbreviation of the airilne has the fewest flights and what country is it in ? ||| flight_2
SQL:  select t1.abbreviation ,  t1.country from airlines as t1 join flights as t2 on t1.uid  =  t2.airline group by t1.airline order by count(*) limit 1

Question 234:  What are airlines that have some flight departing from airport 'AHD ' ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "ahd"

Question 235:  Which airlines have a flight with source airport AHD ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "ahd"

Question 236:  What are airlines that have flights arriving at airport 'AHD ' ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.destairport  =  "ahd"

Question 237:  Which airlines have a flight with destination airport AHD ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.destairport  =  "ahd"

Question 238:  Find all airlines that have flights from both airports 'APG ' and 'CVO ' . ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "apg" intersect select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "cvo"

Question 239:  Which airlines have departing flights from both APG and CVO airports ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "apg" intersect select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "cvo"

Question 240:  Find all airlines that have flights from airport 'CVO ' but not from 'APG ' . ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "cvo" except select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "apg"

Question 241:  Which airlines have departures from CVO but not from APG airports ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "cvo" except select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline where t2.sourceairport  =  "apg"

Question 242:  Find all airlines that have at least 10 flights . ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline group by t1.airline having count(*)  >  10

Question 243:  Which airlines have at least 10 flights ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline group by t1.airline having count(*)  >  10

Question 244:  Find all airlines that have fewer than 200 flights . ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline group by t1.airline having count(*)  <  200

Question 245:  Which airlines have less than 200 flights ? ||| flight_2
SQL:  select t1.airline from airlines as t1 join flights as t2 on t1.uid  =  t2.airline group by t1.airline having count(*)  <  200

Question 246:  What are flight numbers of Airline `` United Airlines '' ? ||| flight_2
SQL:  select t1.flightno from flights as t1 join airlines as t2 on t2.uid  =  t1.airline where t2.airline  =  "united airlines"

Question 247:  Which flight numbers correspond to United Airlines flights ? ||| flight_2
SQL:  select t1.flightno from flights as t1 join airlines as t2 on t2.uid  =  t1.airline where t2.airline  =  "united airlines"

Question 248:  What are flight numbers of flights departing from Airport `` APG '' ? ||| flight_2
SQL:  select flightno from flights where sourceairport  =  "apg"

Question 249:  Give the flight numbers of flights leaving from APG . ||| flight_2
SQL:  select flightno from flights where sourceairport  =  "apg"

Question 250:  What are flight numbers of flights arriving at Airport `` APG '' ? ||| flight_2
SQL:  select flightno from flights where destairport  =  "apg"

Question 251:  Give the flight numbers of flights landing at APG . ||| flight_2
SQL:  select flightno from flights where destairport  =  "apg"

Question 252:  What are flight numbers of flights departing from City `` Aberdeen `` ? ||| flight_2
SQL:  select t1.flightno from flights as t1 join airports as t2 on t1.sourceairport   =  t2.airportcode where t2.city  =  "aberdeen"

Question 253:  Give the flight numbers of flights leaving from Aberdeen . ||| flight_2
SQL:  select t1.flightno from flights as t1 join airports as t2 on t1.sourceairport   =  t2.airportcode where t2.city  =  "aberdeen"

Question 254:  What are flight numbers of flights arriving at City `` Aberdeen '' ? ||| flight_2
SQL:  select t1.flightno from flights as t1 join airports as t2 on t1.destairport   =  t2.airportcode where t2.city  =  "aberdeen"

Question 255:  Give the flight numbers of flights arriving in Aberdeen . ||| flight_2
SQL:  select t1.flightno from flights as t1 join airports as t2 on t1.destairport   =  t2.airportcode where t2.city  =  "aberdeen"

Question 256:  Find the number of flights landing in the city of Aberdeen or Abilene . ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.destairport  =  t2.airportcode where t2.city  =  "aberdeen" or t2.city  =  "abilene"

Question 257:  How many flights land in Aberdeen or Abilene ? ||| flight_2
SQL:  select count(*) from flights as t1 join airports as t2 on t1.destairport  =  t2.airportcode where t2.city  =  "aberdeen" or t2.city  =  "abilene"

Question 258:  Find the name of airports which do not have any flight in and out . ||| flight_2
SQL:  select airportname from airports where airportcode not in (select sourceairport from flights union select destairport from flights)

Question 259:  Which airports do not have departing or arriving flights ? ||| flight_2
SQL:  select airportname from airports where airportcode not in (select sourceairport from flights union select destairport from flights)

Question 260:  How many employees are there ? ||| employee_hire_evaluation
SQL:  select count(*) from employee

Question 261:  Count the number of employees ||| employee_hire_evaluation
SQL:  select count(*) from employee

Question 262:  Sort employee names by their age in ascending order . ||| employee_hire_evaluation
SQL:  select name from employee order by age

Question 263:  List the names of employees and sort in ascending order of age . ||| employee_hire_evaluation
SQL:  select name from employee order by age

Question 264:  What is the number of employees from each city ? ||| employee_hire_evaluation
SQL:  select count(*) ,  city from employee group by city

Question 265:  Count the number of employees for each city . ||| employee_hire_evaluation
SQL:  select count(*) ,  city from employee group by city

Question 266:  Which cities do more than one employee under age 30 come from ? ||| employee_hire_evaluation
SQL:  select city from employee where age  <  30 group by city having count(*)  >  1

Question 267:  Find the cities that have more than one employee under age 30 . ||| employee_hire_evaluation
SQL:  select city from employee where age  <  30 group by city having count(*)  >  1

Question 268:  Find the number of shops in each location . ||| employee_hire_evaluation
SQL:  select count(*) ,  location from shop group by location

Question 269:  How many shops are there in each location ? ||| employee_hire_evaluation
SQL:  select count(*) ,  location from shop group by location

Question 270:  Find the manager name and district of the shop whose number of products is the largest . ||| employee_hire_evaluation
SQL:  select manager_name ,  district from shop order by number_products desc limit 1

Question 271:  What are the manager name and district of the shop that sells the largest number of products ? ||| employee_hire_evaluation
SQL:  select manager_name ,  district from shop order by number_products desc limit 1

Question 272:  find the minimum and maximum number of products of all stores . ||| employee_hire_evaluation
SQL:  select min(number_products) ,  max(number_products) from shop

Question 273:  What are the minimum and maximum number of products across all the shops ? ||| employee_hire_evaluation
SQL:  select min(number_products) ,  max(number_products) from shop

Question 274:  Return the name , location and district of all shops in descending order of number of products . ||| employee_hire_evaluation
SQL:  select name ,  location ,  district from shop order by number_products desc

Question 275:  Sort all the shops by number products in descending order , and return the name , location and district of each shop . ||| employee_hire_evaluation
SQL:  select name ,  location ,  district from shop order by number_products desc

Question 276:  Find the names of stores whose number products is more than the average number of products . ||| employee_hire_evaluation
SQL:  select name from shop where number_products  >  (select avg(number_products) from shop)

Question 277:  Which shops ' number products is above the average ? Give me the shop names . ||| employee_hire_evaluation
SQL:  select name from shop where number_products  >  (select avg(number_products) from shop)

Question 278:  find the name of employee who was awarded the most times in the evaluation . ||| employee_hire_evaluation
SQL:  select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id group by t2.employee_id order by count(*) desc limit 1

Question 279:  Which employee received the most awards in evaluations ? Give me the employee name . ||| employee_hire_evaluation
SQL:  select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id group by t2.employee_id order by count(*) desc limit 1

Question 280:  Find the name of the employee who got the highest one time bonus . ||| employee_hire_evaluation
SQL:  select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id order by t2.bonus desc limit 1

Question 281:  Which employee received the biggest bonus ? Give me the employee name . ||| employee_hire_evaluation
SQL:  select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id order by t2.bonus desc limit 1

Question 282:  Find the names of employees who never won any award in the evaluation . ||| employee_hire_evaluation
SQL:  select name from employee where employee_id not in (select employee_id from evaluation)

Question 283:  What are the names of the employees who never received any evaluation ? ||| employee_hire_evaluation
SQL:  select name from employee where employee_id not in (select employee_id from evaluation)

Question 284:  What is the name of the shop that is hiring the largest number of employees ? ||| employee_hire_evaluation
SQL:  select t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t1.shop_id order by count(*) desc limit 1

Question 285:  Which shop has the most employees ? Give me the shop name . ||| employee_hire_evaluation
SQL:  select t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t1.shop_id order by count(*) desc limit 1

Question 286:  Find the name of the shops that do not hire any employee . ||| employee_hire_evaluation
SQL:  select name from shop where shop_id not in (select shop_id from hiring)

Question 287:  Which shops run with no employees ? Find the shop names ||| employee_hire_evaluation
SQL:  select name from shop where shop_id not in (select shop_id from hiring)

Question 288:  Find the number of employees hired in each shop ; show the shop name as well . ||| employee_hire_evaluation
SQL:  select count(*) ,  t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t2.name

Question 289:  For each shop , return the number of employees working there and the name of the shop . ||| employee_hire_evaluation
SQL:  select count(*) ,  t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t2.name

Question 290:  What is total bonus given in all evaluations ? ||| employee_hire_evaluation
SQL:  select sum(bonus) from evaluation

Question 291:  Find the total amount of bonus given in all the evaluations . ||| employee_hire_evaluation
SQL:  select sum(bonus) from evaluation

Question 292:  Give me all the information about hiring . ||| employee_hire_evaluation
SQL:  select * from hiring

Question 293:  What is all the information about hiring ? ||| employee_hire_evaluation
SQL:  select * from hiring

Question 294:  Which district has both stores with less than 3000 products and stores with more than 10000 products ? ||| employee_hire_evaluation
SQL:  select district from shop where number_products  <  3000 intersect select district from shop where number_products  >  10000

Question 295:  Find the districts in which there are both shops selling less than 3000 products and shops selling more than 10000 products . ||| employee_hire_evaluation
SQL:  select district from shop where number_products  <  3000 intersect select district from shop where number_products  >  10000

Question 296:  How many different store locations are there ? ||| employee_hire_evaluation
SQL:  select count(distinct location) from shop

Question 297:  Count the number of distinct store locations . ||| employee_hire_evaluation
SQL:  select count(distinct location) from shop

Question 298:  How many documents do we have ? ||| cre_Doc_Template_Mgt
SQL:  select count(*) from documents

Question 299:  Count the number of documents . ||| cre_Doc_Template_Mgt
SQL:  select count(*) from documents

Question 300:  List document IDs , document names , and document descriptions for all documents . ||| cre_Doc_Template_Mgt
SQL:  select document_id ,  document_name ,  document_description from documents

Question 301:  What are the ids , names , and descriptions for all documents ? ||| cre_Doc_Template_Mgt
SQL:  select document_id ,  document_name ,  document_description from documents

Question 302:  What is the document name and template id for document with description with the letter 'w ' in it ? ||| cre_Doc_Template_Mgt
SQL:  select document_name ,  template_id from documents where document_description like "%w%"

Question 303:  Return the names and template ids for documents that contain the letter w in their description . ||| cre_Doc_Template_Mgt
SQL:  select document_name ,  template_id from documents where document_description like "%w%"

Question 304:  What is the document id , template id and description for document named `` Robbin CV '' ? ||| cre_Doc_Template_Mgt
SQL:  select document_id ,  template_id ,  document_description from documents where document_name  =  "robbin cv"

Question 305:  Return the document id , template id , and description for the document with the name Robbin CV . ||| cre_Doc_Template_Mgt
SQL:  select document_id ,  template_id ,  document_description from documents where document_name  =  "robbin cv"

Question 306:  How many different templates do all document use ? ||| cre_Doc_Template_Mgt
SQL:  select count(distinct template_id) from documents

Question 307:  Count the number of different templates used for documents . ||| cre_Doc_Template_Mgt
SQL:  select count(distinct template_id) from documents

Question 308:  How many documents are using the template with type code 'PPT ' ? ||| cre_Doc_Template_Mgt
SQL:  select count(*) from documents as t1 join templates as t2 on t1.template_id  =  t2.template_id where t2.template_type_code  =  'ppt'

Question 309:  Count the number of documents that use the PPT template type . ||| cre_Doc_Template_Mgt
SQL:  select count(*) from documents as t1 join templates as t2 on t1.template_id  =  t2.template_id where t2.template_type_code  =  'ppt'

Question 310:  Show all template ids and number of documents using each template . ||| cre_Doc_Template_Mgt
SQL:  select template_id ,  count(*) from documents group by template_id

Question 311:  What are all different template ids used for documents , and how many times were each of them used ? ||| cre_Doc_Template_Mgt
SQL:  select template_id ,  count(*) from documents group by template_id

Question 312:  What is the id and type code for the template used by the most documents ? ||| cre_Doc_Template_Mgt
SQL:  select t1.template_id ,  t2.template_type_code from documents as t1 join templates as t2 on t1.template_id  =  t2.template_id group by t1.template_id order by count(*) desc limit 1

Question 313:  Return the id and type code of the template that is used for the greatest number of documents . ||| cre_Doc_Template_Mgt
SQL:  select t1.template_id ,  t2.template_type_code from documents as t1 join templates as t2 on t1.template_id  =  t2.template_id group by t1.template_id order by count(*) desc limit 1

Question 314:  Show ids for all templates that are used by more than one document . ||| cre_Doc_Template_Mgt
SQL:  select template_id from documents group by template_id having count(*)  >  1

Question 315:  What are the template ids of any templates used in more than a single document ? ||| cre_Doc_Template_Mgt
SQL:  select template_id from documents group by template_id having count(*)  >  1

Question 316:  Show ids for all templates not used by any document . ||| cre_Doc_Template_Mgt
SQL:  select template_id from templates except select template_id from documents

Question 317:  What are the ids for templates that are not used in any documents ? ||| cre_Doc_Template_Mgt
SQL:  select template_id from templates except select template_id from documents

Question 318:  How many templates do we have ? ||| cre_Doc_Template_Mgt
SQL:  select count(*) from templates

Question 319:  Count the number of templates . ||| cre_Doc_Template_Mgt
SQL:  select count(*) from templates

Question 320:  Show template ids , version numbers , and template type codes for all templates . ||| cre_Doc_Template_Mgt
SQL:  select template_id ,  version_number ,  template_type_code from templates

Question 321:  What are the ids , version numbers , and type codes for each template ? ||| cre_Doc_Template_Mgt
SQL:  select template_id ,  version_number ,  template_type_code from templates

Question 322:  Show all distinct template type codes for all templates . ||| cre_Doc_Template_Mgt
SQL:  select distinct template_type_code from templates

Question 323:  What are the different template type codes ? ||| cre_Doc_Template_Mgt
SQL:  select distinct template_type_code from templates

Question 324:  What are the ids of templates with template type code PP or PPT ? ||| cre_Doc_Template_Mgt
SQL:  select template_id from templates where template_type_code  =  "pp" or template_type_code  =  "ppt"

Question 325:  Return the ids of templates that have the code PP or PPT . ||| cre_Doc_Template_Mgt
SQL:  select template_id from templates where template_type_code  =  "pp" or template_type_code  =  "ppt"

Question 326:  How many templates have template type code CV ? ||| cre_Doc_Template_Mgt
SQL:  select count(*) from templates where template_type_code  =  "cv"

Question 327:  Count the number of templates of the type CV . ||| cre_Doc_Template_Mgt
SQL:  select count(*) from templates where template_type_code  =  "cv"

Question 328:  What is the version number and template type code for the template with version number later than 5 ? ||| cre_Doc_Template_Mgt
SQL:  select version_number ,  template_type_code from templates where version_number  >  5

Question 329:  Return the version numbers and template type codes of templates with a version number greater than 5 . ||| cre_Doc_Template_Mgt
SQL:  select version_number ,  template_type_code from templates where version_number  >  5

Question 330:  Show all template type codes and number of templates for each . ||| cre_Doc_Template_Mgt
SQL:  select template_type_code ,  count(*) from templates group by template_type_code

Question 331:  What are the different template type codes , and how many templates correspond to each ? ||| cre_Doc_Template_Mgt
SQL:  select template_type_code ,  count(*) from templates group by template_type_code

Question 332:  Which template type code has most number of templates ? ||| cre_Doc_Template_Mgt
SQL:  select template_type_code from templates group by template_type_code order by count(*) desc limit 1

Question 333:  Return the type code of the template type that the most templates belong to . ||| cre_Doc_Template_Mgt
SQL:  select template_type_code from templates group by template_type_code order by count(*) desc limit 1

Question 334:  Show all template type codes with less than three templates . ||| cre_Doc_Template_Mgt
SQL:  select template_type_code from templates group by template_type_code having count(*)  <  3

Question 335:  What are the codes of template types that have fewer than 3 templates ? ||| cre_Doc_Template_Mgt
SQL:  select template_type_code from templates group by template_type_code having count(*)  <  3

Question 336:  What the smallest version number and its template type code ? ||| cre_Doc_Template_Mgt
SQL:  select min(version_number) ,  template_type_code from templates

Question 337:  Return the lowest version number , along with its corresponding template type code . ||| cre_Doc_Template_Mgt
SQL:  select min(version_number) ,  template_type_code from templates

Question 338:  What is the template type code of the template used by document with the name `` Data base '' ? ||| cre_Doc_Template_Mgt
SQL:  select t1.template_type_code from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id where t2.document_name  =  "data base"

Question 339:  Return the template type code of the template that is used by a document named Data base . ||| cre_Doc_Template_Mgt
SQL:  select t1.template_type_code from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id where t2.document_name  =  "data base"

Question 340:  Show all document names using templates with template type code BK . ||| cre_Doc_Template_Mgt
SQL:  select t2.document_name from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id where t1.template_type_code  =  "bk"

Question 341:  What are the names of documents that use templates with the code BK ? ||| cre_Doc_Template_Mgt
SQL:  select t2.document_name from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id where t1.template_type_code  =  "bk"

Question 342:  Show all template type codes and the number of documents using each type . ||| cre_Doc_Template_Mgt
SQL:  select t1.template_type_code ,  count(*) from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id group by t1.template_type_code

Question 343:  What are the different template type codes , and how many documents use each type ? ||| cre_Doc_Template_Mgt
SQL:  select t1.template_type_code ,  count(*) from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id group by t1.template_type_code

Question 344:  Which template type code is used by most number of documents ? ||| cre_Doc_Template_Mgt
SQL:  select t1.template_type_code from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id group by t1.template_type_code order by count(*) desc limit 1

Question 345:  Return the code of the template type that is most commonly used in documents . ||| cre_Doc_Template_Mgt
SQL:  select t1.template_type_code from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id group by t1.template_type_code order by count(*) desc limit 1

Question 346:  Show all template type codes that are not used by any document . ||| cre_Doc_Template_Mgt
SQL:  select template_type_code from templates except select template_type_code from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id

Question 347:  What are the codes of template types that are not used for any document ? ||| cre_Doc_Template_Mgt
SQL:  select template_type_code from templates except select template_type_code from templates as t1 join documents as t2 on t1.template_id  =  t2.template_id

Question 348:  Show all template type codes and descriptions . ||| cre_Doc_Template_Mgt
SQL:  select template_type_code ,  template_type_description from ref_template_types

Question 349:  What are the type codes and descriptions for all template types ? ||| cre_Doc_Template_Mgt
SQL:  select template_type_code ,  template_type_description from ref_template_types

Question 350:  What is the template type descriptions for template type code `` AD '' . ||| cre_Doc_Template_Mgt
SQL:  select template_type_description from ref_template_types where template_type_code  =  "ad"

Question 351:  Return the template type description of the template type with the code AD . ||| cre_Doc_Template_Mgt
SQL:  select template_type_description from ref_template_types where template_type_code  =  "ad"

Question 352:  What is the template type code for template type description `` Book '' . ||| cre_Doc_Template_Mgt
SQL:  select template_type_code from ref_template_types where template_type_description  =  "book"

Question 353:  Return the type code of the template type with the description `` Book '' . ||| cre_Doc_Template_Mgt
SQL:  select template_type_code from ref_template_types where template_type_description  =  "book"

Question 354:  What are the distinct template type descriptions for the templates ever used by any document ? ||| cre_Doc_Template_Mgt
SQL:  select distinct t1.template_type_description from ref_template_types as t1 join templates as t2 on t1.template_type_code  = t2.template_type_code join documents as t3 on t2.template_id  =  t3.template_id

Question 355:  Return the different descriptions for templates that have been used in a document . ||| cre_Doc_Template_Mgt
SQL:  select distinct t1.template_type_description from ref_template_types as t1 join templates as t2 on t1.template_type_code  = t2.template_type_code join documents as t3 on t2.template_id  =  t3.template_id

Question 356:  What are the template ids with template type description `` Presentation '' . ||| cre_Doc_Template_Mgt
SQL:  select t2.template_id from ref_template_types as t1 join templates as t2 on t1.template_type_code  = t2.template_type_code where t1.template_type_description  =  "presentation"

Question 357:  Return the ids corresponding to templates with the description 'Presentation ' . ||| cre_Doc_Template_Mgt
SQL:  select t2.template_id from ref_template_types as t1 join templates as t2 on t1.template_type_code  = t2.template_type_code where t1.template_type_description  =  "presentation"

Question 358:  How many paragraphs in total ? ||| cre_Doc_Template_Mgt
SQL:  select count(*) from paragraphs

Question 359:  Count the number of paragraphs . ||| cre_Doc_Template_Mgt
SQL:  select count(*) from paragraphs

Question 360:  How many paragraphs for the document with name 'Summer Show ' ? ||| cre_Doc_Template_Mgt
SQL:  select count(*) from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id where t2.document_name  =  'summer show'

Question 361:  Count the number of paragraphs in the document named 'Summer Show ' . ||| cre_Doc_Template_Mgt
SQL:  select count(*) from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id where t2.document_name  =  'summer show'

Question 362:  Show paragraph details for paragraph with text 'Korea ' . ||| cre_Doc_Template_Mgt
SQL:  select other_details from paragraphs where paragraph_text like 'korea'

Question 363:  What are the details for the paragraph that includes the text 'Korea ' ? ||| cre_Doc_Template_Mgt
SQL:  select other_details from paragraphs where paragraph_text like 'korea'

Question 364:  Show all paragraph ids and texts for the document with name 'Welcome to NY ' . ||| cre_Doc_Template_Mgt
SQL:  select t1.paragraph_id ,   t1.paragraph_text from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id where t2.document_name  =  'welcome to ny'

Question 365:  What are the ids and texts of paragraphs in the document titled 'Welcome to NY ' ? ||| cre_Doc_Template_Mgt
SQL:  select t1.paragraph_id ,   t1.paragraph_text from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id where t2.document_name  =  'welcome to ny'

Question 366:  Show all paragraph texts for the document `` Customer reviews '' . ||| cre_Doc_Template_Mgt
SQL:  select t1.paragraph_text from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id where t2.document_name  =  "customer reviews"

Question 367:  What are the paragraph texts for the document with the name 'Customer reviews ' ? ||| cre_Doc_Template_Mgt
SQL:  select t1.paragraph_text from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id where t2.document_name  =  "customer reviews"

Question 368:  Show all document ids and the number of paragraphs in each document . Order by document id . ||| cre_Doc_Template_Mgt
SQL:  select document_id ,  count(*) from paragraphs group by document_id order by document_id

Question 369:  Return the different document ids along with the number of paragraphs corresponding to each , ordered by id . ||| cre_Doc_Template_Mgt
SQL:  select document_id ,  count(*) from paragraphs group by document_id order by document_id

Question 370:  Show all document ids , names and the number of paragraphs in each document . ||| cre_Doc_Template_Mgt
SQL:  select t1.document_id ,  t2.document_name ,  count(*) from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id group by t1.document_id

Question 371:  What are the ids and names of each document , as well as the number of paragraphs in each ? ||| cre_Doc_Template_Mgt
SQL:  select t1.document_id ,  t2.document_name ,  count(*) from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id group by t1.document_id

Question 372:  List all document ids with at least two paragraphs . ||| cre_Doc_Template_Mgt
SQL:  select document_id from paragraphs group by document_id having count(*)  >=  2

Question 373:  What are the ids of documents that have 2 or more paragraphs ? ||| cre_Doc_Template_Mgt
SQL:  select document_id from paragraphs group by document_id having count(*)  >=  2

Question 374:  What is the document id and name with greatest number of paragraphs ? ||| cre_Doc_Template_Mgt
SQL:  select t1.document_id ,  t2.document_name from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id group by t1.document_id order by count(*) desc limit 1

Question 375:  Return the id and name of the document with the most paragraphs . ||| cre_Doc_Template_Mgt
SQL:  select t1.document_id ,  t2.document_name from paragraphs as t1 join documents as t2 on t1.document_id  =  t2.document_id group by t1.document_id order by count(*) desc limit 1

Question 376:  What is the document id with least number of paragraphs ? ||| cre_Doc_Template_Mgt
SQL:  select document_id from paragraphs group by document_id order by count(*) asc limit 1

Question 377:  Return the id of the document with the fewest paragraphs . ||| cre_Doc_Template_Mgt
SQL:  select document_id from paragraphs group by document_id order by count(*) asc limit 1

Question 378:  What is the document id with 1 to 2 paragraphs ? ||| cre_Doc_Template_Mgt
SQL:  select document_id from paragraphs group by document_id having count(*) between 1 and 2

Question 379:  Give the ids of documents that have between one and two paragraphs . ||| cre_Doc_Template_Mgt
SQL:  select document_id from paragraphs group by document_id having count(*) between 1 and 2

Question 380:  Show the document id with paragraph text 'Brazil ' and 'Ireland ' . ||| cre_Doc_Template_Mgt
SQL:  select document_id from paragraphs where paragraph_text  =  'brazil' intersect select document_id from paragraphs where paragraph_text  =  'ireland'

Question 381:  What are the ids of documents that contain the paragraph text 'Brazil ' and 'Ireland ' ? ||| cre_Doc_Template_Mgt
SQL:  select document_id from paragraphs where paragraph_text  =  'brazil' intersect select document_id from paragraphs where paragraph_text  =  'ireland'

Question 382:  How many teachers are there ? ||| course_teach
SQL:  select count(*) from teacher

Question 383:  What is the total count of teachers ? ||| course_teach
SQL:  select count(*) from teacher

Question 384:  List the names of teachers in ascending order of age . ||| course_teach
SQL:  select name from teacher order by age asc

Question 385:  What are the names of the teachers ordered by ascending age ? ||| course_teach
SQL:  select name from teacher order by age asc

Question 386:  What are the age and hometown of teachers ? ||| course_teach
SQL:  select age ,  hometown from teacher

Question 387:  What is the age and hometown of every teacher ? ||| course_teach
SQL:  select age ,  hometown from teacher

Question 388:  List the name of teachers whose hometown is not `` Little Lever Urban District '' . ||| course_teach
SQL:  select name from teacher where hometown != "little lever urban district"

Question 389:  What are the names of the teachers whose hometown is not `` Little Lever Urban District '' ? ||| course_teach
SQL:  select name from teacher where hometown != "little lever urban district"

Question 390:  Show the name of teachers aged either 32 or 33 ? ||| course_teach
SQL:  select name from teacher where age  =  32 or age  =  33

Question 391:  What are the names of the teachers who are aged either 32 or 33 ? ||| course_teach
SQL:  select name from teacher where age  =  32 or age  =  33

Question 392:  What is the hometown of the youngest teacher ? ||| course_teach
SQL:  select hometown from teacher order by age asc limit 1

Question 393:  Where is the youngest teacher from ? ||| course_teach
SQL:  select hometown from teacher order by age asc limit 1

Question 394:  Show different hometown of teachers and the number of teachers from each hometown . ||| course_teach
SQL:  select hometown ,  count(*) from teacher group by hometown

Question 395:  For each hometown , how many teachers are there ? ||| course_teach
SQL:  select hometown ,  count(*) from teacher group by hometown

Question 396:  List the most common hometown of teachers . ||| course_teach
SQL:  select hometown from teacher group by hometown order by count(*) desc limit 1

Question 397:  What is the most commmon hometowns for teachers ? ||| course_teach
SQL:  select hometown from teacher group by hometown order by count(*) desc limit 1

Question 398:  Show the hometowns shared by at least two teachers . ||| course_teach
SQL:  select hometown from teacher group by hometown having count(*)  >=  2

Question 399:  What are the towns from which at least two teachers come from ? ||| course_teach
SQL:  select hometown from teacher group by hometown having count(*)  >=  2

Question 400:  Show names of teachers and the courses they are arranged to teach . ||| course_teach
SQL:  select t3.name ,  t2.course from course_arrange as t1 join course as t2 on t1.course_id  =  t2.course_id join teacher as t3 on t1.teacher_id  =  t3.teacher_id

Question 401:  What is the name of each teacher and what course they teach ? ||| course_teach
SQL:  select t3.name ,  t2.course from course_arrange as t1 join course as t2 on t1.course_id  =  t2.course_id join teacher as t3 on t1.teacher_id  =  t3.teacher_id

Question 402:  Show names of teachers and the courses they are arranged to teach in ascending alphabetical order of the teacher 's name . ||| course_teach
SQL:  select t3.name ,  t2.course from course_arrange as t1 join course as t2 on t1.course_id  =  t2.course_id join teacher as t3 on t1.teacher_id  =  t3.teacher_id order by t3.name

Question 403:  What are the names of the teachers and the courses they teach in ascending alphabetical order by the name of the teacher ? ||| course_teach
SQL:  select t3.name ,  t2.course from course_arrange as t1 join course as t2 on t1.course_id  =  t2.course_id join teacher as t3 on t1.teacher_id  =  t3.teacher_id order by t3.name

Question 404:  Show the name of the teacher for the math course . ||| course_teach
SQL:  select t3.name from course_arrange as t1 join course as t2 on t1.course_id  =  t2.course_id join teacher as t3 on t1.teacher_id  =  t3.teacher_id where t2.course  =  "math"

Question 405:  What are the names of the people who teach math courses ? ||| course_teach
SQL:  select t3.name from course_arrange as t1 join course as t2 on t1.course_id  =  t2.course_id join teacher as t3 on t1.teacher_id  =  t3.teacher_id where t2.course  =  "math"

Question 406:  Show names of teachers and the number of courses they teach . ||| course_teach
SQL:  select t2.name ,  count(*) from course_arrange as t1 join teacher as t2 on t1.teacher_id  =  t2.teacher_id group by t2.name

Question 407:  What are the names of the teachers and how many courses do they teach ? ||| course_teach
SQL:  select t2.name ,  count(*) from course_arrange as t1 join teacher as t2 on t1.teacher_id  =  t2.teacher_id group by t2.name

Question 408:  Show names of teachers that teach at least two courses . ||| course_teach
SQL:  select t2.name from course_arrange as t1 join teacher as t2 on t1.teacher_id  =  t2.teacher_id group by t2.name having count(*)  >=  2

Question 409:  What are the names of the teachers who teach at least two courses ? ||| course_teach
SQL:  select t2.name from course_arrange as t1 join teacher as t2 on t1.teacher_id  =  t2.teacher_id group by t2.name having count(*)  >=  2

Question 410:  List the names of teachers who have not been arranged to teach courses . ||| course_teach
SQL:  select name from teacher where teacher_id not in (select teacher_id from course_arrange)

Question 411:  What are the names of the teachers whose courses have not been arranged ? ||| course_teach
SQL:  select name from teacher where teacher_id not in (select teacher_id from course_arrange)

Question 412:  How many visitors below age 30 are there ? ||| museum_visit
SQL:  select count(*) from visitor where age  <  30

Question 413:  Find the names of the visitors whose membership level is higher than 4 , and order the results by the level from high to low . ||| museum_visit
SQL:  select name from visitor where level_of_membership  >  4 order by level_of_membership desc

Question 414:  What is the average age of the visitors whose membership level is not higher than 4 ? ||| museum_visit
SQL:  select avg(age) from visitor where level_of_membership  <=  4

Question 415:  Find the name and membership level of the visitors whose membership level is higher than 4 , and sort by their age from old to young . ||| museum_visit
SQL:  select name ,  level_of_membership from visitor where level_of_membership  >  4 order by age desc

Question 416:  Find the id and name of the museum that has the most staff members ? ||| museum_visit
SQL:  select museum_id ,  name from museum order by num_of_staff desc limit 1

Question 417:  Find the average number of staff working for the museums that were open before 2009 . ||| museum_visit
SQL:  select avg(num_of_staff) from museum where open_year  <  2009

Question 418:  What are the opening year and staff number of the museum named Plaza Museum ? ||| museum_visit
SQL:  select num_of_staff ,  open_year from museum where name  =  'plaza museum'

Question 419:  find the names of museums which have more staff than the minimum staff number of all museums opened after 2010 . ||| museum_visit
SQL:  select name from museum where num_of_staff  >  (select min(num_of_staff) from museum where open_year  >  2010)

Question 420:  find the id , name and age for visitors who visited some museums more than once . ||| museum_visit
SQL:  select t1.id ,  t1.name ,  t1.age from visitor as t1 join visit as t2 on t1.id  =  t2.visitor_id group by t1.id having count(*)  >  1

Question 421:  What are the id , name and membership level of visitors who have spent the largest amount of money in total in all museum tickets ? ||| museum_visit
SQL:  select t2.visitor_id ,  t1.name ,  t1.level_of_membership from visitor as t1 join visit as t2 on t1.id  =  t2.visitor_id group by t2.visitor_id order by sum(t2.total_spent) desc limit 1

Question 422:  What are the id and name of the museum visited most times ? ||| museum_visit
SQL:  select t2.museum_id ,  t1.name from museum as t1 join visit as t2 on t1.museum_id  =  t2.museum_id group by t2.museum_id order by count(*) desc limit 1

Question 423:  What is the name of the museum that had no visitor yet ? ||| museum_visit
SQL:  select name from museum where museum_id not in (select museum_id from visit)

Question 424:  Find the name and age of the visitor who bought the most tickets at once . ||| museum_visit
SQL:  select t1.name ,  t1.age from visitor as t1 join visit as t2 on t1.id  =  t2.visitor_id order by t2.num_of_ticket desc limit 1

Question 425:  What are the average and maximum number of tickets bought in all visits ? ||| museum_visit
SQL:  select avg(num_of_ticket) ,  max(num_of_ticket) from visit

Question 426:  What is the total ticket expense of the visitors whose membership level is 1 ? ||| museum_visit
SQL:  select sum(t2.total_spent) from visitor as t1 join visit as t2 on t1.id  =  t2.visitor_id where t1.level_of_membership  =  1

Question 427:  What is the name of the visitor who visited both a museum opened before 2009 and a museum opened after 2011 ? ||| museum_visit
SQL:  select t1.name from visitor as t1 join visit as t2 on t1.id  =  t2.visitor_id join museum as t3 on t3.museum_id  =  t2.museum_id where t3.open_year  <  2009 intersect select t1.name from visitor as t1 join visit as t2 on t1.id  =  t2.visitor_id join museum as t3 on t3.museum_id  =  t2.museum_id where t3.open_year  >  2011

Question 428:  Find the number of visitors who did not visit any museum opened after 2010 . ||| museum_visit
SQL:  select count(*) from visitor where id not in (select t2.visitor_id from museum as t1 join visit as t2 on t1.museum_id  =  t2.museum_id where t1.open_year  >  2010)

Question 429:  How many museums were opened after 2013 or before 2008 ? ||| museum_visit
SQL:  select count(*) from museum where open_year  >  2013 or open_year  <  2008

Question 430:  Find the total number of players . ||| wta_1
SQL:  select count(*) from players

Question 431:  How many players are there ? ||| wta_1
SQL:  select count(*) from players

Question 432:  Find the total number of matches . ||| wta_1
SQL:  select count(*) from matches

Question 433:  Count the number of matches . ||| wta_1
SQL:  select count(*) from matches

Question 434:  List the first name and birth date of all players from the country with code USA . ||| wta_1
SQL:  select first_name ,  birth_date from players where country_code  =  'usa'

Question 435:  What are the first names and birth dates of players from the USA ? ||| wta_1
SQL:  select first_name ,  birth_date from players where country_code  =  'usa'

Question 436:  Find the average age of losers and winners of all matches . ||| wta_1
SQL:  select avg(loser_age) ,  avg(winner_age) from matches

Question 437:  What are the average ages of losers and winners across matches ? ||| wta_1
SQL:  select avg(loser_age) ,  avg(winner_age) from matches

Question 438:  Find the average rank of winners in all matches . ||| wta_1
SQL:  select avg(winner_rank) from matches

Question 439:  What is the average rank for winners in all matches ? ||| wta_1
SQL:  select avg(winner_rank) from matches

Question 440:  Find the highest rank of losers in all matches . ||| wta_1
SQL:  select min(loser_rank) from matches

Question 441:  What is the best rank of losers across all matches ? ||| wta_1
SQL:  select min(loser_rank) from matches

Question 442:  find the number of distinct country codes of all players . ||| wta_1
SQL:  select count(distinct country_code) from players

Question 443:  How many distinct countries do players come from ? ||| wta_1
SQL:  select count(distinct country_code) from players

Question 444:  Find the number of distinct name of losers . ||| wta_1
SQL:  select count(distinct loser_name) from matches

Question 445:  How many different loser names are there ? ||| wta_1
SQL:  select count(distinct loser_name) from matches

Question 446:  Find the name of tourney that has more than 10 matches . ||| wta_1
SQL:  select tourney_name from matches group by tourney_name having count(*)  >  10

Question 447:  What are the names of tournaments that have more than 10 matches ? ||| wta_1
SQL:  select tourney_name from matches group by tourney_name having count(*)  >  10

Question 448:  List the names of all winners who played in both 2013 and 2016 . ||| wta_1
SQL:  select winner_name from matches where year  =  2013 intersect select winner_name from matches where year  =  2016

Question 449:  What are the names of players who won in both 2013 and 2016 ? ||| wta_1
SQL:  select winner_name from matches where year  =  2013 intersect select winner_name from matches where year  =  2016

Question 450:  List the number of all matches who played in years of 2013 or 2016 . ||| wta_1
SQL:  select count(*) from matches where year  =  2013 or year  =  2016

Question 451:  How many matches were played in 2013 or 2016 ? ||| wta_1
SQL:  select count(*) from matches where year  =  2013 or year  =  2016

Question 452:  What are the country code and first name of the players who won in both tourney WTA Championships and Australian Open ? ||| wta_1
SQL:  select t1.country_code ,  t1.first_name from players as t1 join matches as t2 on t1.player_id  =  t2.winner_id where t2.tourney_name  =  'wta championships' intersect select t1.country_code ,  t1.first_name from players as t1 join matches as t2 on t1.player_id  =  t2.winner_id where t2.tourney_name  =  'australian open'

Question 453:  What are the first names and country codes for players who won both the WTA Championships and the Australian Open ? ||| wta_1
SQL:  select t1.country_code ,  t1.first_name from players as t1 join matches as t2 on t1.player_id  =  t2.winner_id where t2.tourney_name  =  'wta championships' intersect select t1.country_code ,  t1.first_name from players as t1 join matches as t2 on t1.player_id  =  t2.winner_id where t2.tourney_name  =  'australian open'

Question 454:  Find the first name and country code of the oldest player . ||| wta_1
SQL:  select first_name ,  country_code from players order by birth_date limit 1

Question 455:  What is the first name and country code of the oldest player ? ||| wta_1
SQL:  select first_name ,  country_code from players order by birth_date limit 1

Question 456:  List the first and last name of all players in the order of birth date . ||| wta_1
SQL:  select first_name ,  last_name from players order by birth_date

Question 457:  What are the full names of all players , sorted by birth date ? ||| wta_1
SQL:  select first_name ,  last_name from players order by birth_date

Question 458:  List the first and last name of all players who are left / L hand in the order of birth date . ||| wta_1
SQL:  select first_name ,  last_name from players where hand  =  'l' order by birth_date

Question 459:  What are the full names of all left handed players , in order of birth date ? ||| wta_1
SQL:  select first_name ,  last_name from players where hand  =  'l' order by birth_date

Question 460:  Find the first name and country code of the player who did the most number of tours . ||| wta_1
SQL:  select t1.country_code ,  t1.first_name from players as t1 join rankings as t2 on t1.player_id  =  t2.player_id order by t2.tours desc limit 1

Question 461:  What is the first name and country code of the player with the most tours ? ||| wta_1
SQL:  select t1.country_code ,  t1.first_name from players as t1 join rankings as t2 on t1.player_id  =  t2.player_id order by t2.tours desc limit 1

Question 462:  Find the year that has the most number of matches . ||| wta_1
SQL:  select year from matches group by year order by count(*) desc limit 1

Question 463:  Which year had the most matches ? ||| wta_1
SQL:  select year from matches group by year order by count(*) desc limit 1

Question 464:  Find the name and rank points of the winner who won the most times . ||| wta_1
SQL:  select winner_name ,  winner_rank_points from matches group by winner_name order by count(*) desc limit 1

Question 465:  What is the name of the winner who has won the most matches , and how many rank points does this player have ? ||| wta_1
SQL:  select winner_name ,  winner_rank_points from matches group by winner_name order by count(*) desc limit 1

Question 466:  Find the name of the winner who has the highest rank points and participated in the Australian Open tourney . ||| wta_1
SQL:  select winner_name from matches where tourney_name  =  'australian open' order by winner_rank_points desc limit 1

Question 467:  What is the name of the winner with the most rank points who participated in the Australian Open tournament ? ||| wta_1
SQL:  select winner_name from matches where tourney_name  =  'australian open' order by winner_rank_points desc limit 1

Question 468:  find the names of loser and winner who played in the match with greatest number of minutes . ||| wta_1
SQL:  select winner_name ,  loser_name from matches order by minutes desc limit 1

Question 469:  What are the names of the winner and loser who played in the longest match ? ||| wta_1
SQL:  select winner_name ,  loser_name from matches order by minutes desc limit 1

Question 470:  Find the average ranking for each player and their first name . ||| wta_1
SQL:  select avg(ranking) ,  t1.first_name from players as t1 join rankings as t2 on t1.player_id  =  t2.player_id group by t1.first_name

Question 471:  What are the first names of all players , and their average rankings ? ||| wta_1
SQL:  select avg(ranking) ,  t1.first_name from players as t1 join rankings as t2 on t1.player_id  =  t2.player_id group by t1.first_name

Question 472:  Find the total ranking points for each player and their first name . ||| wta_1
SQL:  select sum(ranking_points) ,  t1.first_name from players as t1 join rankings as t2 on t1.player_id  =  t2.player_id group by t1.first_name

Question 473:  What are the first names of all players , and their total ranking points ? ||| wta_1
SQL:  select sum(ranking_points) ,  t1.first_name from players as t1 join rankings as t2 on t1.player_id  =  t2.player_id group by t1.first_name

Question 474:  find the number of players for each country . ||| wta_1
SQL:  select count(*) ,  country_code from players group by country_code

Question 475:  How many players are from each country ? ||| wta_1
SQL:  select count(*) ,  country_code from players group by country_code

Question 476:  find the code of the country where has the greatest number of players . ||| wta_1
SQL:  select country_code from players group by country_code order by count(*) desc limit 1

Question 477:  What is the code of the country with the most players ? ||| wta_1
SQL:  select country_code from players group by country_code order by count(*) desc limit 1

Question 478:  Find the codes of countries that have more than 50 players . ||| wta_1
SQL:  select country_code from players group by country_code having count(*)  >  50

Question 479:  What are the codes of countries with more than 50 players ? ||| wta_1
SQL:  select country_code from players group by country_code having count(*)  >  50

Question 480:  Find the total number of tours for each ranking date . ||| wta_1
SQL:  select sum(tours) ,  ranking_date from rankings group by ranking_date

Question 481:  How many total tours were there for each ranking date ? ||| wta_1
SQL:  select sum(tours) ,  ranking_date from rankings group by ranking_date

Question 482:  Find the number of matches happened in each year . ||| wta_1
SQL:  select count(*) ,  year from matches group by year

Question 483:  How many matches were played in each year ? ||| wta_1
SQL:  select count(*) ,  year from matches group by year

Question 484:  Find the name and rank of the 3 youngest winners across all matches . ||| wta_1
SQL:  select distinct winner_name ,  winner_rank from matches order by winner_age limit 3

Question 485:  What are the names and ranks of the three youngest winners across all matches ? ||| wta_1
SQL:  select distinct winner_name ,  winner_rank from matches order by winner_age limit 3

Question 486:  How many different winners both participated in the WTA Championships and were left handed ? ||| wta_1
SQL:  select count(distinct winner_name) from matches where tourney_name  =  'wta championships' and winner_hand  =  'l'

Question 487:  Find the number of left handed winners who participated in the WTA Championships . ||| wta_1
SQL:  select count(distinct winner_name) from matches where tourney_name  =  'wta championships' and winner_hand  =  'l'

Question 488:  Find the first name , country code and birth date of the winner who has the highest rank points in all matches . ||| wta_1
SQL:  select t1.first_name ,  t1.country_code ,  t1.birth_date from players as t1 join matches as t2 on t1.player_id  =  t2.winner_id order by t2.winner_rank_points desc limit 1

Question 489:  What is the first name , country code , and birth date of the player with the most winner rank points across all matches ? ||| wta_1
SQL:  select t1.first_name ,  t1.country_code ,  t1.birth_date from players as t1 join matches as t2 on t1.player_id  =  t2.winner_id order by t2.winner_rank_points desc limit 1

Question 490:  Find the number of players for each hand type . ||| wta_1
SQL:  select count(*) ,  hand from players group by hand

Question 491:  How many players are there for each hand type ? ||| wta_1
SQL:  select count(*) ,  hand from players group by hand

Question 492:  How many ships ended up being 'Captured ' ? ||| battle_death
SQL:  select count(*) from ship where disposition_of_ship  =  'captured'

Question 493:  List the name and tonnage ordered by in descending alphaetical order for the names . ||| battle_death
SQL:  select name ,  tonnage from ship order by name desc

Question 494:  List the name , date and result of each battle . ||| battle_death
SQL:  select name ,  date from battle

Question 495:  What is maximum and minimum death toll caused each time ? ||| battle_death
SQL:  select max(killed) ,  min(killed) from death

Question 496:  What is the average number of injuries caused each time ? ||| battle_death
SQL:  select avg(injured) from death

Question 497:  What are the death and injury situations caused by the ship with tonnage 't ' ? ||| battle_death
SQL:  select t1.killed ,  t1.injured from death as t1 join ship as t2 on t1.caused_by_ship_id  =  t2.id where t2.tonnage  =  't'

Question 498:  What are the name and results of the battles when the bulgarian commander is not 'Boril ' ||| battle_death
SQL:  select name ,  result from battle where bulgarian_commander != 'boril'

Question 499:  What are the different ids and names of the battles that lost any 'Brig ' type shipes ? ||| battle_death
SQL:  select distinct t1.id ,  t1.name from battle as t1 join ship as t2 on t1.id  =  t2.lost_in_battle where t2.ship_type  =  'brig'

Question 500:  What are the ids and names of the battles that led to more than 10 people killed in total . ||| battle_death
SQL:  select t1.id ,  t1.name from battle as t1 join ship as t2 on t1.id  =  t2.lost_in_battle join death as t3 on t2.id  =  t3.caused_by_ship_id group by t1.id having sum(t3.killed)  >  10

Question 501:  What is the ship id and name that caused most total injuries ? ||| battle_death
SQL:  select t2.id ,  t2.name from death as t1 join ship as t2 on t1.caused_by_ship_id  =  t2.id group by t2.id order by count(*) desc limit 1

Question 502:  What are the distinct battle names which are between bulgarian commander 'Kaloyan ' and latin commander 'Baldwin I ' ? ||| battle_death
SQL:  select name from battle where bulgarian_commander  =  'kaloyan' and latin_commander  =  'baldwin i'

Question 503:  How many different results are there for the battles ? ||| battle_death
SQL:  select count(distinct result) from battle

Question 504:  How many battles did not lose any ship with tonnage '225 ' ? ||| battle_death
SQL:  select count(*) from battle where id not in ( select lost_in_battle from ship where tonnage  =  '225' );

Question 505:  List the name and date the battle that has lost the ship named 'Lettice ' and the ship named 'HMS Atalanta ' ||| battle_death
SQL:  select t1.name ,  t1.date from battle as t1 join ship as t2 on t1.id  =  t2.lost_in_battle where t2.name  =  'lettice' intersect select t1.name ,  t1.date from battle as t1 join ship as t2 on t1.id  =  t2.lost_in_battle where t2.name  =  'hms atalanta'

Question 506:  Show names , results and bulgarian commanders of the battles with no ships lost in the 'English Channel ' . ||| battle_death
SQL:  select name ,  result ,  bulgarian_commander from battle except select t1.name ,  t1.result ,  t1.bulgarian_commander from battle as t1 join ship as t2 on t1.id  =  t2.lost_in_battle where t2.location  =  'english channel'

Question 507:  What are the notes of the death events which has substring 'East ' ? ||| battle_death
SQL:  select note from death where note like '%east%'

Question 508:  what are all the addresses including line 1 and line 2 ? ||| student_transcripts_tracking
SQL:  select line_1 ,  line_2 from addresses

Question 509:  What is the first and second line for all addresses ? ||| student_transcripts_tracking
SQL:  select line_1 ,  line_2 from addresses

Question 510:  How many courses in total are listed ? ||| student_transcripts_tracking
SQL:  select count(*) from courses

Question 511:  How many courses are there ? ||| student_transcripts_tracking
SQL:  select count(*) from courses

Question 512:  How is the math course described ? ||| student_transcripts_tracking
SQL:  select course_description from courses where course_name  =  'math'

Question 513:  What are the descriptions for all the math courses ? ||| student_transcripts_tracking
SQL:  select course_description from courses where course_name  =  'math'

Question 514:  What is the zip code of the address in the city Port Chelsea ? ||| student_transcripts_tracking
SQL:  select zip_postcode from addresses where city  =  'port chelsea'

Question 515:  What is the zip code for Port Chelsea ? ||| student_transcripts_tracking
SQL:  select zip_postcode from addresses where city  =  'port chelsea'

Question 516:  Which department offers the most number of degrees ? List department name and id . ||| student_transcripts_tracking
SQL:  select t2.department_name ,  t1.department_id from degree_programs as t1 join departments as t2 on t1.department_id  =  t2.department_id group by t1.department_id order by count(*) desc limit 1

Question 517:  What is the name and id of the department with the most number of degrees ? ||| student_transcripts_tracking
SQL:  select t2.department_name ,  t1.department_id from degree_programs as t1 join departments as t2 on t1.department_id  =  t2.department_id group by t1.department_id order by count(*) desc limit 1

Question 518:  How many departments offer any degree ? ||| student_transcripts_tracking
SQL:  select count(distinct department_id) from degree_programs

Question 519:  How many different departments offer degrees ? ||| student_transcripts_tracking
SQL:  select count(distinct department_id) from degree_programs

Question 520:  How many different degree names are offered ? ||| student_transcripts_tracking
SQL:  select count(distinct degree_summary_name) from degree_programs

Question 521:  How many different degrees are offered ? ||| student_transcripts_tracking
SQL:  select count(distinct degree_summary_name) from degree_programs

Question 522:  How many degrees does the engineering department offer ? ||| student_transcripts_tracking
SQL:  select count(*) from departments as t1 join degree_programs as t2 on t1.department_id  =  t2.department_id where t1.department_name  =  'engineer'

Question 523:  How many degrees does the engineering department have ? ||| student_transcripts_tracking
SQL:  select count(*) from departments as t1 join degree_programs as t2 on t1.department_id  =  t2.department_id where t1.department_name  =  'engineer'

Question 524:  What are the names and descriptions of all the sections ? ||| student_transcripts_tracking
SQL:  select section_name ,  section_description from sections

Question 525:  What are the names and descriptions for all the sections ? ||| student_transcripts_tracking
SQL:  select section_name ,  section_description from sections

Question 526:  What are the names and id of courses having at most 2 sections ? ||| student_transcripts_tracking
SQL:  select t1.course_name ,  t1.course_id from courses as t1 join sections as t2 on t1.course_id  =  t2.course_id group by t1.course_id having count(*)  <=  2

Question 527:  What are the names and ids of every course with less than 2 sections ? ||| student_transcripts_tracking
SQL:  select t1.course_name ,  t1.course_id from courses as t1 join sections as t2 on t1.course_id  =  t2.course_id group by t1.course_id having count(*)  <=  2

Question 528:  List the section_name in reversed lexicographical order . ||| student_transcripts_tracking
SQL:  select section_name from sections order by section_name desc

Question 529:  What are the names of the sections in reverse alphabetical order ? ||| student_transcripts_tracking
SQL:  select section_name from sections order by section_name desc

Question 530:  What is the semester which most student registered in ? Show both the name and the id . ||| student_transcripts_tracking
SQL:  select t1.semester_name ,  t1.semester_id from semesters as t1 join student_enrolment as t2 on t1.semester_id  =  t2.semester_id group by t1.semester_id order by count(*) desc limit 1

Question 531:  For each semester , what is the name and id of the one with the most students registered ? ||| student_transcripts_tracking
SQL:  select t1.semester_name ,  t1.semester_id from semesters as t1 join student_enrolment as t2 on t1.semester_id  =  t2.semester_id group by t1.semester_id order by count(*) desc limit 1

Question 532:  What is the description of the department whose name has the substring the computer ? ||| student_transcripts_tracking
SQL:  select department_description from departments where department_name like '%computer%'

Question 533:  What is the department description for the one whose name has the word computer ? ||| student_transcripts_tracking
SQL:  select department_description from departments where department_name like '%computer%'

Question 534:  Who are enrolled in 2 degree programs in one semester ? List the first name , middle name and last name and the id . ||| student_transcripts_tracking
SQL:  select t1.first_name ,  t1.middle_name ,  t1.last_name ,  t1.student_id from students as t1 join student_enrolment as t2 on t1.student_id  =  t2.student_id group by t1.student_id having count(*)  =  2

Question 535:  What are the first , middle , and last names , along with the ids , of all students who enrolled in 2 degree programs in one semester ? ||| student_transcripts_tracking
SQL:  select t1.first_name ,  t1.middle_name ,  t1.last_name ,  t1.student_id from students as t1 join student_enrolment as t2 on t1.student_id  =  t2.student_id group by t1.student_id having count(*)  =  2

Question 536:  Who is enrolled in a Bachelor degree program ? List the first name , middle name , last name . ||| student_transcripts_tracking
SQL:  select distinct t1.first_name ,  t1.middle_name ,  t1.last_name from students as t1 join student_enrolment as t2 on t1.student_id  =  t2.student_id join degree_programs as t3 on t2.degree_program_id  =  t3.degree_program_id where t3.degree_summary_name  =  'bachelor'

Question 537:  What are the first , middle , and last names for everybody enrolled in a Bachelors program ? ||| student_transcripts_tracking
SQL:  select distinct t1.first_name ,  t1.middle_name ,  t1.last_name from students as t1 join student_enrolment as t2 on t1.student_id  =  t2.student_id join degree_programs as t3 on t2.degree_program_id  =  t3.degree_program_id where t3.degree_summary_name  =  'bachelor'

Question 538:  Find the kind of program which most number of students are enrolled in ? ||| student_transcripts_tracking
SQL:  select t1.degree_summary_name from degree_programs as t1 join student_enrolment as t2 on t1.degree_program_id  =  t2.degree_program_id group by t1.degree_summary_name order by count(*) desc limit 1

Question 539:  What is the degree summary name that has the most number of students enrolled ? ||| student_transcripts_tracking
SQL:  select t1.degree_summary_name from degree_programs as t1 join student_enrolment as t2 on t1.degree_program_id  =  t2.degree_program_id group by t1.degree_summary_name order by count(*) desc limit 1

Question 540:  Find the program which most number of students are enrolled in . List both the id and the summary . ||| student_transcripts_tracking
SQL:  select t1.degree_program_id ,  t1.degree_summary_name from degree_programs as t1 join student_enrolment as t2 on t1.degree_program_id  =  t2.degree_program_id group by t1.degree_program_id order by count(*) desc limit 1

Question 541:  What is the program id and the summary of the degree that has the most students enrolled ? ||| student_transcripts_tracking
SQL:  select t1.degree_program_id ,  t1.degree_summary_name from degree_programs as t1 join student_enrolment as t2 on t1.degree_program_id  =  t2.degree_program_id group by t1.degree_program_id order by count(*) desc limit 1

Question 542:  Which student has enrolled for the most times in any program ? List the id , first name , middle name , last name , the number of enrollments and student id . ||| student_transcripts_tracking
SQL:  select t1.student_id ,  t1.first_name ,  t1.middle_name ,  t1.last_name ,  count(*) ,  t1.student_id from students as t1 join student_enrolment as t2 on t1.student_id  =  t2.student_id group by t1.student_id order by count(*) desc limit 1

Question 543:  What is the first , middle , and last name , along with the id and number of enrollments , for the student who enrolled the most in any program ? ||| student_transcripts_tracking
SQL:  select t1.student_id ,  t1.first_name ,  t1.middle_name ,  t1.last_name ,  count(*) ,  t1.student_id from students as t1 join student_enrolment as t2 on t1.student_id  =  t2.student_id group by t1.student_id order by count(*) desc limit 1

Question 544:  Which semesters do not have any student enrolled ? List the semester name . ||| student_transcripts_tracking
SQL:  select semester_name from semesters where semester_id not in( select semester_id from student_enrolment )

Question 545:  What is the name of the semester with no students enrolled ? ||| student_transcripts_tracking
SQL:  select semester_name from semesters where semester_id not in( select semester_id from student_enrolment )

Question 546:  What are all the course names of the courses which ever have students enrolled in ? ||| student_transcripts_tracking
SQL:  select distinct t1.course_name from courses as t1 join student_enrolment_courses as t2 on t1.course_id  =  t2.course_id

Question 547:  What are the names of all courses that have some students enrolled ? ||| student_transcripts_tracking
SQL:  select distinct t1.course_name from courses as t1 join student_enrolment_courses as t2 on t1.course_id  =  t2.course_id

Question 548:  What 's the name of the course with most number of enrollments ? ||| student_transcripts_tracking
SQL:  select  t1.course_name from courses as t1 join student_enrolment_courses as t2 on t1.course_id  =  t2.course_id group by t1.course_name order by count(*) desc limit 1

Question 549:  What is the name of the course with the most students enrolled ? ||| student_transcripts_tracking
SQL:  select  t1.course_name from courses as t1 join student_enrolment_courses as t2 on t1.course_id  =  t2.course_id group by t1.course_name order by count(*) desc limit 1

Question 550:  Find the last name of the students who currently live in the state of North Carolina but have not registered in any degree program . ||| student_transcripts_tracking
SQL:  select t1.last_name from students as t1 join addresses as t2 on t1.current_address_id  =  t2.address_id where t2.state_province_county  =  'northcarolina' except select distinct t3.last_name from students as t3 join student_enrolment as t4 on t3.student_id  =  t4.student_id

Question 551:  What are the last name of the students who live in North Carolina but have not registered in any degree programs ? ||| student_transcripts_tracking
SQL:  select t1.last_name from students as t1 join addresses as t2 on t1.current_address_id  =  t2.address_id where t2.state_province_county  =  'northcarolina' except select distinct t3.last_name from students as t3 join student_enrolment as t4 on t3.student_id  =  t4.student_id

Question 552:  Show the date and id of the transcript with at least 2 course results . ||| student_transcripts_tracking
SQL:  select t2.transcript_date ,  t1.transcript_id from transcript_contents as t1 join transcripts as t2 on t1.transcript_id  =  t2.transcript_id group by t1.transcript_id having count(*)  >=  2

Question 553:  What is the date and id of the transcript with at least 2 courses listed ? ||| student_transcripts_tracking
SQL:  select t2.transcript_date ,  t1.transcript_id from transcript_contents as t1 join transcripts as t2 on t1.transcript_id  =  t2.transcript_id group by t1.transcript_id having count(*)  >=  2

Question 554:  What is the phone number of the man with the first name Timmothy and the last name Ward ? ||| student_transcripts_tracking
SQL:  select cell_mobile_number from students where first_name  =  'timmothy' and last_name  =  'ward'

Question 555:  What is the mobile phone number of the student named Timmothy Ward ? ||| student_transcripts_tracking
SQL:  select cell_mobile_number from students where first_name  =  'timmothy' and last_name  =  'ward'

Question 556:  Who is the first student to register ? List the first name , middle name and last name . ||| student_transcripts_tracking
SQL:  select first_name ,  middle_name ,  last_name from students order by date_first_registered asc limit 1

Question 557:  What is the first , middle , and last name of the first student to register ? ||| student_transcripts_tracking
SQL:  select first_name ,  middle_name ,  last_name from students order by date_first_registered asc limit 1

Question 558:  Who is the earliest graduate of the school ? List the first name , middle name and last name . ||| student_transcripts_tracking
SQL:  select first_name ,  middle_name ,  last_name from students order by date_left asc limit 1

Question 559:  What is the first , middle , and last name of the earliest school graduate ? ||| student_transcripts_tracking
SQL:  select first_name ,  middle_name ,  last_name from students order by date_left asc limit 1

Question 560:  Whose permanent address is different from his or her current address ? List his or her first name . ||| student_transcripts_tracking
SQL:  select first_name from students where current_address_id != permanent_address_id

Question 561:  What is the first name of the student whose permanent address is different from his or her current one ? ||| student_transcripts_tracking
SQL:  select first_name from students where current_address_id != permanent_address_id

Question 562:  Which address holds the most number of students currently ? List the address id and all lines . ||| student_transcripts_tracking
SQL:  select t1.address_id ,  t1.line_1 ,  t1.line_2 from addresses as t1 join students as t2 on t1.address_id  =  t2.current_address_id group by t1.address_id order by count(*) desc limit 1

Question 563:  What is the id , line 1 , and line 2 of the address with the most students ? ||| student_transcripts_tracking
SQL:  select t1.address_id ,  t1.line_1 ,  t1.line_2 from addresses as t1 join students as t2 on t1.address_id  =  t2.current_address_id group by t1.address_id order by count(*) desc limit 1

Question 564:  On average , when were the transcripts printed ? ||| student_transcripts_tracking
SQL:  select avg(transcript_date) from transcripts

Question 565:  What is the average transcript date ? ||| student_transcripts_tracking
SQL:  select avg(transcript_date) from transcripts

Question 566:  When is the first transcript released ? List the date and details . ||| student_transcripts_tracking
SQL:  select transcript_date ,  other_details from transcripts order by transcript_date asc limit 1

Question 567:  What is the earliest date of a transcript release , and what details can you tell me ? ||| student_transcripts_tracking
SQL:  select transcript_date ,  other_details from transcripts order by transcript_date asc limit 1

Question 568:  How many transcripts are released ? ||| student_transcripts_tracking
SQL:  select count(*) from transcripts

Question 569:  How many transcripts are listed ? ||| student_transcripts_tracking
SQL:  select count(*) from transcripts

Question 570:  What is the last transcript release date ? ||| student_transcripts_tracking
SQL:  select transcript_date from transcripts order by transcript_date desc limit 1

Question 571:  When was the last transcript released ? ||| student_transcripts_tracking
SQL:  select transcript_date from transcripts order by transcript_date desc limit 1

Question 572:  How many times at most can a course enrollment result show in different transcripts ? Also show the course enrollment id . ||| student_transcripts_tracking
SQL:  select count(*) ,  student_course_id from transcript_contents group by student_course_id order by count(*) desc limit 1

Question 573:  What is the maximum number of times that a course shows up in different transcripts and what is that course 's enrollment id ? ||| student_transcripts_tracking
SQL:  select count(*) ,  student_course_id from transcript_contents group by student_course_id order by count(*) desc limit 1

Question 574:  Show the date of the transcript which shows the least number of results , also list the id . ||| student_transcripts_tracking
SQL:  select t2.transcript_date ,  t1.transcript_id from transcript_contents as t1 join transcripts as t2 on t1.transcript_id  =  t2.transcript_id group by t1.transcript_id order by count(*) asc limit 1

Question 575:  What is the date and id of the transcript with the least number of results ? ||| student_transcripts_tracking
SQL:  select t2.transcript_date ,  t1.transcript_id from transcript_contents as t1 join transcripts as t2 on t1.transcript_id  =  t2.transcript_id group by t1.transcript_id order by count(*) asc limit 1

Question 576:  Find the semester when both Master students and Bachelor students got enrolled in . ||| student_transcripts_tracking
SQL:  select distinct t2.semester_id from degree_programs as t1 join student_enrolment as t2 on t1.degree_program_id  =  t2.degree_program_id where degree_summary_name  =  'master' intersect select distinct t2.semester_id from degree_programs as t1 join student_enrolment as t2 on t1.degree_program_id  =  t2.degree_program_id where degree_summary_name  =  'bachelor'

Question 577:  What is the id of the semester that had both Masters and Bachelors students enrolled ? ||| student_transcripts_tracking
SQL:  select distinct t2.semester_id from degree_programs as t1 join student_enrolment as t2 on t1.degree_program_id  =  t2.degree_program_id where degree_summary_name  =  'master' intersect select distinct t2.semester_id from degree_programs as t1 join student_enrolment as t2 on t1.degree_program_id  =  t2.degree_program_id where degree_summary_name  =  'bachelor'

Question 578:  How many different addresses do the students currently live ? ||| student_transcripts_tracking
SQL:  select count(distinct current_address_id) from students

Question 579:  What are the different addresses that have students living there ? ||| student_transcripts_tracking
SQL:  select count(distinct current_address_id) from students

Question 580:  List all the student details in reversed lexicographical order . ||| student_transcripts_tracking
SQL:  select other_student_details from students order by other_student_details desc

Question 581:  What other details can you tell me about students in reverse alphabetical order ? ||| student_transcripts_tracking
SQL:  select other_student_details from students order by other_student_details desc

Question 582:  Describe the section h . ||| student_transcripts_tracking
SQL:  select section_description from sections where section_name  =  'h'

Question 583:  What is the description for the section named h ? ||| student_transcripts_tracking
SQL:  select section_description from sections where section_name  =  'h'

Question 584:  Find the first name of the students who permanently live in the country Haiti or have the cell phone number 09700166582 . ||| student_transcripts_tracking
SQL:  select t1.first_name from students as t1 join addresses as t2 on t1.permanent_address_id  =  t2.address_id where t2.country  =  'haiti' or t1.cell_mobile_number  =  '09700166582'

Question 585:  What are the first names of the students who live in Haiti permanently or have the cell phone number 09700166582 ? ||| student_transcripts_tracking
SQL:  select t1.first_name from students as t1 join addresses as t2 on t1.permanent_address_id  =  t2.address_id where t2.country  =  'haiti' or t1.cell_mobile_number  =  '09700166582'

Question 586:  List the title of all cartoons in alphabetical order . ||| tvshow
SQL:  select title from cartoon order by title

Question 587:  What are the titles of the cartoons sorted alphabetically ? ||| tvshow
SQL:  select title from cartoon order by title

Question 588:  List all cartoon directed by `` Ben Jones '' . ||| tvshow
SQL:  select title from cartoon where directed_by = "ben jones";

Question 589:  What are the names of all cartoons directed by Ben Jones ? ||| tvshow
SQL:  select title from cartoon where directed_by = "ben jones";

Question 590:  How many cartoons were written by `` Joseph Kuhr '' ? ||| tvshow
SQL:  select count(*) from cartoon where written_by = "joseph kuhr";

Question 591:  What is the number of cartoones written by Joseph Kuhr ? ||| tvshow
SQL:  select count(*) from cartoon where written_by = "joseph kuhr";

Question 592:  list all cartoon titles and their directors ordered by their air date ||| tvshow
SQL:  select title ,  directed_by from cartoon order by original_air_date

Question 593:  What is the name and directors of all the cartoons that are ordered by air date ? ||| tvshow
SQL:  select title ,  directed_by from cartoon order by original_air_date

Question 594:  List the title of all cartoon directed by `` Ben Jones '' or `` Brandon Vietti '' . ||| tvshow
SQL:  select title from cartoon where directed_by = "ben jones" or directed_by = "brandon vietti";

Question 595:  What are the titles of all cartoons directed by Ben Jones or Brandon Vietti ? ||| tvshow
SQL:  select title from cartoon where directed_by = "ben jones" or directed_by = "brandon vietti";

Question 596:  Which country has the most of TV Channels ? List the country and number of TV Channels it has . ||| tvshow
SQL:  select country ,  count(*) from tv_channel group by country order by count(*) desc limit 1;

Question 597:  What is the country with the most number of TV Channels and how many does it have ? ||| tvshow
SQL:  select country ,  count(*) from tv_channel group by country order by count(*) desc limit 1;

Question 598:  List the number of different series names and contents in the TV Channel table . ||| tvshow
SQL:  select count(distinct series_name) ,  count(distinct content) from tv_channel;

Question 599:  How many different series and contents are listed in the TV Channel table ? ||| tvshow
SQL:  select count(distinct series_name) ,  count(distinct content) from tv_channel;

Question 600:  What is the content of TV Channel with serial name `` Sky Radio '' ? ||| tvshow
SQL:  select content from tv_channel where series_name = "sky radio";

Question 601:  What is the content of the series Sky Radio ? ||| tvshow
SQL:  select content from tv_channel where series_name = "sky radio";

Question 602:  What is the Package Option of TV Channel with serial name `` Sky Radio '' ? ||| tvshow
SQL:  select package_option from tv_channel where series_name = "sky radio";

Question 603:  What are the Package Options of the TV Channels whose series names are Sky Radio ? ||| tvshow
SQL:  select package_option from tv_channel where series_name = "sky radio";

Question 604:  How many TV Channel using language English ? ||| tvshow
SQL:  select count(*) from tv_channel where language = "english";

Question 605:  How many TV Channels use the English language ? ||| tvshow
SQL:  select count(*) from tv_channel where language = "english";

Question 606:  List the language used least number of TV Channel . List language and number of TV Channel . ||| tvshow
SQL:  select language ,  count(*) from tv_channel group by language order by count(*) asc limit 1;

Question 607:  What are the languages used by the least number of TV Channels and how many channels use it ? ||| tvshow
SQL:  select language ,  count(*) from tv_channel group by language order by count(*) asc limit 1;

Question 608:  List each language and the number of TV Channels using it . ||| tvshow
SQL:  select language ,  count(*) from tv_channel group by language

Question 609:  For each language , list the number of TV Channels that use it . ||| tvshow
SQL:  select language ,  count(*) from tv_channel group by language

Question 610:  What is the TV Channel that shows the cartoon `` The Rise of the Blue Beetle ! '' ? List the TV Channel 's series name . ||| tvshow
SQL:  select t1.series_name from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.title = "the rise of the blue beetle!";

Question 611:  What is the series name of the TV Channel that shows the cartoon `` The Rise of the Blue Beetle '' ? ||| tvshow
SQL:  select t1.series_name from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.title = "the rise of the blue beetle!";

Question 612:  List the title of all Cartoons showed on TV Channel with series name `` Sky Radio '' . ||| tvshow
SQL:  select t2.title from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t1.series_name = "sky radio";

Question 613:  What is the title of all the cartools that are on the TV Channel with the series name `` Sky Radio '' ? ||| tvshow
SQL:  select t2.title from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t1.series_name = "sky radio";

Question 614:  List the Episode of all TV series sorted by rating . ||| tvshow
SQL:  select episode from tv_series order by rating

Question 615:  What are all of the episodes ordered by ratings ? ||| tvshow
SQL:  select episode from tv_series order by rating

Question 616:  List top 3 highest Rating TV series . List the TV series 's Episode and Rating . ||| tvshow
SQL:  select episode ,  rating from tv_series order by rating desc limit 3;

Question 617:  What are 3 most highly rated episodes in the TV series table and what were those ratings ? ||| tvshow
SQL:  select episode ,  rating from tv_series order by rating desc limit 3;

Question 618:  What is minimum and maximum share of TV series ? ||| tvshow
SQL:  select max(share) , min(share) from tv_series;

Question 619:  What is the maximum and minimum share for the TV series ? ||| tvshow
SQL:  select max(share) , min(share) from tv_series;

Question 620:  What is the air date of TV series with Episode `` A Love of a Lifetime '' ? ||| tvshow
SQL:  select air_date from tv_series where episode = "a love of a lifetime";

Question 621:  When did the episode `` A Love of a Lifetime '' air ? ||| tvshow
SQL:  select air_date from tv_series where episode = "a love of a lifetime";

Question 622:  What is Weekly Rank of TV series with Episode `` A Love of a Lifetime '' ? ||| tvshow
SQL:  select weekly_rank from tv_series where episode = "a love of a lifetime";

Question 623:  What is the weekly rank for the episode `` A Love of a Lifetime '' ? ||| tvshow
SQL:  select weekly_rank from tv_series where episode = "a love of a lifetime";

Question 624:  What is the TV Channel of TV series with Episode `` A Love of a Lifetime '' ? List the TV Channel 's series name . ||| tvshow
SQL:  select t1.series_name from tv_channel as t1 join tv_series as t2 on t1.id = t2.channel where t2.episode = "a love of a lifetime";

Question 625:  What is the name of the series that has the episode `` A Love of a Lifetime '' ? ||| tvshow
SQL:  select t1.series_name from tv_channel as t1 join tv_series as t2 on t1.id = t2.channel where t2.episode = "a love of a lifetime";

Question 626:  List the Episode of all TV series showed on TV Channel with series name `` Sky Radio '' . ||| tvshow
SQL:  select t2.episode from tv_channel as t1 join tv_series as t2 on t1.id = t2.channel where t1.series_name = "sky radio";

Question 627:  What is the episode for the TV series named `` Sky Radio '' ? ||| tvshow
SQL:  select t2.episode from tv_channel as t1 join tv_series as t2 on t1.id = t2.channel where t1.series_name = "sky radio";

Question 628:  Find the number of cartoons directed by each of the listed directors . ||| tvshow
SQL:  select count(*) ,  directed_by from cartoon group by directed_by

Question 629:  How many cartoons did each director create ? ||| tvshow
SQL:  select count(*) ,  directed_by from cartoon group by directed_by

Question 630:  Find the production code and channel of the most recently aired cartoon . ||| tvshow
SQL:  select production_code ,  channel from cartoon order by original_air_date desc limit 1

Question 631:  What is the produdction code and channel of the most recent cartoon ? ||| tvshow
SQL:  select production_code ,  channel from cartoon order by original_air_date desc limit 1

Question 632:  Find the package choice and series name of the TV channel that has high definition TV . ||| tvshow
SQL:  select package_option ,  series_name from tv_channel where hight_definition_tv  =  "yes"

Question 633:  What are the package options and the name of the series for the TV Channel that supports high definition TV ? ||| tvshow
SQL:  select package_option ,  series_name from tv_channel where hight_definition_tv  =  "yes"

Question 634:  which countries ' tv channels are playing some cartoon written by Todd Casey ? ||| tvshow
SQL:  select t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.written_by  =  'todd casey'

Question 635:  What are the countries that have cartoons on TV that were written by Todd Casey ? ||| tvshow
SQL:  select t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.written_by  =  'todd casey'

Question 636:  which countries ' tv channels are not playing any cartoon written by Todd Casey ? ||| tvshow
SQL:  select country from tv_channel except select t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.written_by  =  'todd casey'

Question 637:  What are the countries that are not playing cartoons written by Todd Casey ? ||| tvshow
SQL:  select country from tv_channel except select t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.written_by  =  'todd casey'

Question 638:  Find the series name and country of the tv channel that is playing some cartoons directed by Ben Jones and Michael Chang ? ||| tvshow
SQL:  select t1.series_name ,  t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by  =  'michael chang' intersect select t1.series_name ,  t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by  =  'ben jones'

Question 639:  What is the series name and country of all TV channels that are playing cartoons directed by Ben Jones and cartoons directed by Michael Chang ? ||| tvshow
SQL:  select t1.series_name ,  t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by  =  'michael chang' intersect select t1.series_name ,  t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by  =  'ben jones'

Question 640:  find the pixel aspect ratio and nation of the tv channels that do not use English . ||| tvshow
SQL:  select pixel_aspect_ratio_par ,  country from tv_channel where language != 'english'

Question 641:  What is the pixel aspect ratio and country of origin for all TV channels that do not use English ? ||| tvshow
SQL:  select pixel_aspect_ratio_par ,  country from tv_channel where language != 'english'

Question 642:  find id of the tv channels that from the countries where have more than two tv channels . ||| tvshow
SQL:  select id from tv_channel group by country having count(*)  >  2

Question 643:  What are the ids of all tv channels that have more than 2 TV channels ? ||| tvshow
SQL:  select id from tv_channel group by country having count(*)  >  2

Question 644:  find the id of tv channels that do not play any cartoon directed by Ben Jones . ||| tvshow
SQL:  select id from tv_channel except select channel from cartoon where directed_by  =  'ben jones'

Question 645:  What are the ids of the TV channels that do not have any cartoons directed by Ben Jones ? ||| tvshow
SQL:  select id from tv_channel except select channel from cartoon where directed_by  =  'ben jones'

Question 646:  find the package option of the tv channel that do not have any cartoon directed by Ben Jones . ||| tvshow
SQL:  select package_option from tv_channel where id not in (select channel from cartoon where directed_by  =  'ben jones')

Question 647:  What are the package options of all tv channels that are not playing any cartoons directed by Ben Jones ? ||| tvshow
SQL:  select package_option from tv_channel where id not in (select channel from cartoon where directed_by  =  'ben jones')

Question 648:  How many poker players are there ? ||| poker_player
SQL:  select count(*) from poker_player

Question 649:  Count the number of poker players . ||| poker_player
SQL:  select count(*) from poker_player

Question 650:  List the earnings of poker players in descending order . ||| poker_player
SQL:  select earnings from poker_player order by earnings desc

Question 651:  What are the earnings of poker players , ordered descending by value ? ||| poker_player
SQL:  select earnings from poker_player order by earnings desc

Question 652:  List the final tables made and the best finishes of poker players . ||| poker_player
SQL:  select final_table_made ,  best_finish from poker_player

Question 653:  What are the final tables made and best finishes for all poker players ? ||| poker_player
SQL:  select final_table_made ,  best_finish from poker_player

Question 654:  What is the average earnings of poker players ? ||| poker_player
SQL:  select avg(earnings) from poker_player

Question 655:  Return the average earnings across all poker players . ||| poker_player
SQL:  select avg(earnings) from poker_player

Question 656:  What is the money rank of the poker player with the highest earnings ? ||| poker_player
SQL:  select money_rank from poker_player order by earnings desc limit 1

Question 657:  Return the money rank of the player with the greatest earnings . ||| poker_player
SQL:  select money_rank from poker_player order by earnings desc limit 1

Question 658:  What is the maximum number of final tables made among poker players with earnings less than 200000 ? ||| poker_player
SQL:  select max(final_table_made) from poker_player where earnings  <  200000

Question 659:  Return the maximum final tables made across all poker players who have earnings below 200000 . ||| poker_player
SQL:  select max(final_table_made) from poker_player where earnings  <  200000

Question 660:  What are the names of poker players ? ||| poker_player
SQL:  select t1.name from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id

Question 661:  Return the names of all the poker players . ||| poker_player
SQL:  select t1.name from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id

Question 662:  What are the names of poker players whose earnings is higher than 300000 ? ||| poker_player
SQL:  select t1.name from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id where t2.earnings  >  300000

Question 663:  Give the names of poker players who have earnings above 300000 . ||| poker_player
SQL:  select t1.name from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id where t2.earnings  >  300000

Question 664:  List the names of poker players ordered by the final tables made in ascending order . ||| poker_player
SQL:  select t1.name from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id order by t2.final_table_made

Question 665:  What are the names of poker players , ordered ascending by the number of final tables they have made ? ||| poker_player
SQL:  select t1.name from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id order by t2.final_table_made

Question 666:  What is the birth date of the poker player with the lowest earnings ? ||| poker_player
SQL:  select t1.birth_date from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id order by t2.earnings asc limit 1

Question 667:  Return the birth date of the poker player with the lowest earnings . ||| poker_player
SQL:  select t1.birth_date from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id order by t2.earnings asc limit 1

Question 668:  What is the money rank of the tallest poker player ? ||| poker_player
SQL:  select t2.money_rank from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id order by t1.height desc limit 1

Question 669:  Return the money rank of the poker player with the greatest height . ||| poker_player
SQL:  select t2.money_rank from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id order by t1.height desc limit 1

Question 670:  What is the average earnings of poker players with height higher than 200 ? ||| poker_player
SQL:  select avg(t2.earnings) from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id where t1.height  >  200

Question 671:  Give average earnings of poker players who are taller than 200 . ||| poker_player
SQL:  select avg(t2.earnings) from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id where t1.height  >  200

Question 672:  What are the names of poker players in descending order of earnings ? ||| poker_player
SQL:  select t1.name from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id order by t2.earnings desc

Question 673:  Return the names of poker players sorted by their earnings descending . ||| poker_player
SQL:  select t1.name from people as t1 join poker_player as t2 on t1.people_id  =  t2.people_id order by t2.earnings desc

Question 674:  What are different nationalities of people and the corresponding number of people from each nation ? ||| poker_player
SQL:  select nationality ,  count(*) from people group by nationality

Question 675:  How many people are there of each nationality ? ||| poker_player
SQL:  select nationality ,  count(*) from people group by nationality

Question 676:  What is the most common nationality of people ? ||| poker_player
SQL:  select nationality from people group by nationality order by count(*) desc limit 1

Question 677:  Give the nationality that is most common across all people . ||| poker_player
SQL:  select nationality from people group by nationality order by count(*) desc limit 1

Question 678:  What are the nationalities that are shared by at least two people ? ||| poker_player
SQL:  select nationality from people group by nationality having count(*)  >=  2

Question 679:  Return the nationalities for which there are two or more people . ||| poker_player
SQL:  select nationality from people group by nationality having count(*)  >=  2

Question 680:  List the names and birth dates of people in ascending alphabetical order of name . ||| poker_player
SQL:  select name ,  birth_date from people order by name asc

Question 681:  What are the names and birth dates of people , ordered by their names in alphabetical order ? ||| poker_player
SQL:  select name ,  birth_date from people order by name asc

Question 682:  Show names of people whose nationality is not `` Russia '' . ||| poker_player
SQL:  select name from people where nationality != "russia"

Question 683:  What are the names of people who are not from Russia ? ||| poker_player
SQL:  select name from people where nationality != "russia"

Question 684:  List the names of people that are not poker players . ||| poker_player
SQL:  select name from people where people_id not in (select people_id from poker_player)

Question 685:  What are the names of people who do not play poker ? ||| poker_player
SQL:  select name from people where people_id not in (select people_id from poker_player)

Question 686:  How many distinct nationalities are there ? ||| poker_player
SQL:  select count(distinct nationality) from people

Question 687:  Count the number of different nationalities . ||| poker_player
SQL:  select count(distinct nationality) from people

Question 688:  How many states are there ? ||| voter_1
SQL:  select count(*) from area_code_state

Question 689:  List the contestant numbers and names , ordered by contestant name descending . ||| voter_1
SQL:  select contestant_number ,  contestant_name from contestants order by contestant_name desc

Question 690:  List the vote ids , phone numbers and states of all votes . ||| voter_1
SQL:  select vote_id ,  phone_number ,  state from votes

Question 691:  What are the maximum and minimum values of area codes ? ||| voter_1
SQL:  select max(area_code) ,  min(area_code) from area_code_state

Question 692:  What is last date created of votes from the state 'CA ' ? ||| voter_1
SQL:  select max(created) from votes where state  =  'ca'

Question 693:  What are the names of the contestants whose names are not 'Jessie Alloway ' ||| voter_1
SQL:  select contestant_name from contestants where contestant_name != 'jessie alloway'

Question 694:  What are the distinct states and create time of all votes ? ||| voter_1
SQL:  select distinct state ,  created from votes

Question 695:  What are the contestant numbers and names of the contestants who had at least two votes ? ||| voter_1
SQL:  select t1.contestant_number , t1.contestant_name from contestants as t1 join votes as t2 on t1.contestant_number  =  t2.contestant_number group by t1.contestant_number having count(*)  >=  2

Question 696:  Of all the contestants who got voted , what is the contestant number and name of the contestant who got least votes ? ||| voter_1
SQL:  select t1.contestant_number , t1.contestant_name from contestants as t1 join votes as t2 on t1.contestant_number  =  t2.contestant_number group by t1.contestant_number order by count(*) asc limit 1

Question 697:  What are the number of votes from state 'NY ' or 'CA ' ? ||| voter_1
SQL:  select count(*) from votes where state  =  'ny' or state  =  'ca'

Question 698:  How many contestants did not get voted ? ||| voter_1
SQL:  select count(*) from contestants where contestant_number not in ( select contestant_number from votes )

Question 699:  What is the area code in which the most voters voted ? ||| voter_1
SQL:  select t1.area_code from area_code_state as t1 join votes as t2 on t1.state  =  t2.state group by t1.area_code order by count(*) desc limit 1

Question 700:  What are the create dates , states , and phone numbers of the votes that were for the contestant named 'Tabatha Gehling ' ? ||| voter_1
SQL:  select t2.created ,  t2.state ,  t2.phone_number from contestants as t1 join votes as t2 on t1.contestant_number  =  t2.contestant_number where t1.contestant_name  =  'tabatha gehling'

Question 701:  List the area codes in which voters voted both for the contestant 'Tabatha Gehling ' and the contestant 'Kelly Clauss ' . ||| voter_1
SQL:  select t3.area_code from contestants as t1 join votes as t2 on t1.contestant_number  =  t2.contestant_number join area_code_state as t3 on t2.state  =  t3.state where t1.contestant_name  =  'tabatha gehling' intersect select t3.area_code from contestants as t1 join votes as t2 on t1.contestant_number  =  t2.contestant_number join area_code_state as t3 on t2.state  =  t3.state where t1.contestant_name  =  'kelly clauss'

Question 702:  Return the names of the contestants whose names contain the substring 'Al' . ||| voter_1
SQL:  select contestant_name from contestants where contestant_name like "%al%"

Question 703:  What are the names of all the countries that became independent after 1950 ? ||| world_1
SQL:  select name from country where indepyear  >  1950

Question 704:  Give the names of the nations that were founded after 1950 . ||| world_1
SQL:  select name from country where indepyear  >  1950

Question 705:  How many countries have a republic as their form of government ? ||| world_1
SQL:  select count(*) from country where governmentform  =  "republic"

Question 706:  How many countries have governments that are republics ? ||| world_1
SQL:  select count(*) from country where governmentform  =  "republic"

Question 707:  What is the total surface area of the countries in the Caribbean region ? ||| world_1
SQL:  select sum(surfacearea) from country where region  =  "caribbean"

Question 708:  How much surface area do the countires in the Carribean cover together ? ||| world_1
SQL:  select sum(surfacearea) from country where region  =  "caribbean"

Question 709:  Which continent is Anguilla in ? ||| world_1
SQL:  select continent from country where name  =  "anguilla"

Question 710:  What is the continent name which Anguilla belongs to ? ||| world_1
SQL:  select continent from country where name  =  "anguilla"

Question 711:  Which region is the city Kabul located in ? ||| world_1
SQL:  select region from country as t1 join city as t2 on t1.code  =  t2.countrycode where t2.name  =  "kabul"

Question 712:  What region is Kabul in ? ||| world_1
SQL:  select region from country as t1 join city as t2 on t1.code  =  t2.countrycode where t2.name  =  "kabul"

Question 713:  Which language is the most popular in Aruba ? ||| world_1
SQL:  select t2.language from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.name  =  "aruba" order by percentage desc limit 1

Question 714:  What language is predominantly spoken in Aruba ? ||| world_1
SQL:  select t2.language from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.name  =  "aruba" order by percentage desc limit 1

Question 715:  What are the population and life expectancies in Brazil ? ||| world_1
SQL:  select population ,  lifeexpectancy from country where name  =  "brazil"

Question 716:  Give me Brazil’s population and life expectancies . ||| world_1
SQL:  select population ,  lifeexpectancy from country where name  =  "brazil"

Question 717:  What are the region and population of Angola ? ||| world_1
SQL:  select population ,  region from country where name  =  "angola"

Question 718:  What region does Angola belong to and what is its population ? ||| world_1
SQL:  select population ,  region from country where name  =  "angola"

Question 719:  What is the average expected life expectancy for countries in the region of Central Africa ? ||| world_1
SQL:  select avg(lifeexpectancy) from country where region  =  "central africa"

Question 720:  How long is the people’s average life expectancy in Central Africa ? ||| world_1
SQL:  select avg(lifeexpectancy) from country where region  =  "central africa"

Question 721:  What is the name of country that has the shortest life expectancy in Asia ? ||| world_1
SQL:  select name from country where continent  =  "asia" order by lifeexpectancy limit 1

Question 722:  Give the name of the country in Asia with the lowest life expectancy . ||| world_1
SQL:  select name from country where continent  =  "asia" order by lifeexpectancy limit 1

Question 723:  What is the total population and maximum GNP in Asia ? ||| world_1
SQL:  select sum(population) ,  max(gnp) from country where continent  =  "asia"

Question 724:  How many people live in Asia , and what is the largest GNP among them ? ||| world_1
SQL:  select sum(population) ,  max(gnp) from country where continent  =  "asia"

Question 725:  What is the average life expectancy in African countries that are republics ? ||| world_1
SQL:  select avg(lifeexpectancy) from country where continent  =  "africa" and governmentform  =  "republic"

Question 726:  Give the average life expectancy for countries in Africa which are republics ? ||| world_1
SQL:  select avg(lifeexpectancy) from country where continent  =  "africa" and governmentform  =  "republic"

Question 727:  What is the total surface area of the continents Asia and Europe ? ||| world_1
SQL:  select sum(surfacearea) from country where continent  =  "asia" or continent  =  "europe"

Question 728:  Give the total surface area covered by countries in Asia or Europe . ||| world_1
SQL:  select sum(surfacearea) from country where continent  =  "asia" or continent  =  "europe"

Question 729:  How many people live in Gelderland district ? ||| world_1
SQL:  select sum(population) from city where district  =  "gelderland"

Question 730:  What is the total population of Gelderland district ? ||| world_1
SQL:  select sum(population) from city where district  =  "gelderland"

Question 731:  What is the average GNP and total population in all nations whose government is US territory ? ||| world_1
SQL:  select avg(gnp) ,  sum(population) from country where governmentform  =  "us territory"

Question 732:  Give the mean GNP and total population of nations which are considered US territory . ||| world_1
SQL:  select avg(gnp) ,  sum(population) from country where governmentform  =  "us territory"

Question 733:  How many unique languages are spoken in the world ? ||| world_1
SQL:  select count(distinct language) from countrylanguage

Question 734:  What is the number of distinct languages used around the world ? ||| world_1
SQL:  select count(distinct language) from countrylanguage

Question 735:  How many type of governments are in Africa ? ||| world_1
SQL:  select count(distinct governmentform) from country where continent  =  "africa"

Question 736:  How many different forms of governments are there in Africa ? ||| world_1
SQL:  select count(distinct governmentform) from country where continent  =  "africa"

Question 737:  What is the total number of languages used in Aruba ? ||| world_1
SQL:  select count(t2.language) from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.name  =  "aruba"

Question 738:  How many languages are spoken in Aruba ? ||| world_1
SQL:  select count(t2.language) from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.name  =  "aruba"

Question 739:  How many official languages does Afghanistan have ? ||| world_1
SQL:  select count(*) from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.name  =  "afghanistan" and isofficial  =  "t"

Question 740:  How many official languages are spoken in Afghanistan ? ||| world_1
SQL:  select count(*) from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.name  =  "afghanistan" and isofficial  =  "t"

Question 741:  What is name of the country that speaks the largest number of languages ? ||| world_1
SQL:  select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode group by t1.name order by count(*) desc limit 1

Question 742:  Give the name of the nation that uses the greatest amount of languages . ||| world_1
SQL:  select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode group by t1.name order by count(*) desc limit 1

Question 743:  Which continent has the most diverse languages ? ||| world_1
SQL:  select t1.continent from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode group by t1.continent order by count(*) desc limit 1

Question 744:  Which continent speaks the most languages ? ||| world_1
SQL:  select t1.continent from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode group by t1.continent order by count(*) desc limit 1

Question 745:  How many countries speak both English and Dutch ? ||| world_1
SQL:  select count(*) from (select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "dutch")

Question 746:  What is the number of nations that use English and Dutch ? ||| world_1
SQL:  select count(*) from (select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "dutch")

Question 747:  What are the names of nations speak both English and French ? ||| world_1
SQL:  select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "french"

Question 748:  Give the names of nations that speak both English and French . ||| world_1
SQL:  select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "french"

Question 749:  What are the names of nations where both English and French are official languages ? ||| world_1
SQL:  select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" and t2.isofficial  =  "t" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "french" and t2.isofficial  =  "t"

Question 750:  Give the names of countries with English and French as official languages . ||| world_1
SQL:  select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" and t2.isofficial  =  "t" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "french" and t2.isofficial  =  "t"

Question 751:  What is the number of distinct continents where Chinese is spoken ? ||| world_1
SQL:  select count( distinct continent) from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "chinese"

Question 752:  How many continents speak Chinese ? ||| world_1
SQL:  select count( distinct continent) from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "chinese"

Question 753:  What are the regions that use English or Dutch ? ||| world_1
SQL:  select distinct t1.region from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" or t2.language  =  "dutch"

Question 754:  Which regions speak Dutch or English ? ||| world_1
SQL:  select distinct t1.region from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" or t2.language  =  "dutch"

Question 755:  What are the countries where either English or Dutch is the official language ? ||| world_1
SQL:  select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" and isofficial  =  "t" union select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "dutch" and isofficial  =  "t"

Question 756:  Which countries have either English or Dutch as an official language ? ||| world_1
SQL:  select * from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" and isofficial  =  "t" union select * from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "dutch" and isofficial  =  "t"

Question 757:  Which language is the most popular on the Asian continent ? ||| world_1
SQL:  select t2.language from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.continent  =  "asia" group by t2.language order by count (*) desc limit 1

Question 758:  What is the language that is used by the largest number of Asian nations ? ||| world_1
SQL:  select t2.language from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.continent  =  "asia" group by t2.language order by count (*) desc limit 1

Question 759:  Which languages are spoken by only one country in republic governments ? ||| world_1
SQL:  select t2.language from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.governmentform  =  "republic" group by t2.language having count(*)  =  1

Question 760:  What languages are only used by a single country with a republic government ? ||| world_1
SQL:  select t2.language from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.governmentform  =  "republic" group by t2.language having count(*)  =  1

Question 761:  Find the city with the largest population that uses English . ||| world_1
SQL:  select t1.name ,  t1.population from city as t1 join countrylanguage as t2 on t1.countrycode  =  t2.countrycode where t2.language  =  "english" order by t1.population desc limit 1

Question 762:  What is the most populace city that speaks English ? ||| world_1
SQL:  select t1.name ,  t1.population from city as t1 join countrylanguage as t2 on t1.countrycode  =  t2.countrycode where t2.language  =  "english" order by t1.population desc limit 1

Question 763:  Find the name , population and expected life length of asian country with the largest area ? ||| world_1
SQL:  select name ,  population ,  lifeexpectancy from country where continent  =  "asia" order by surfacearea desc limit 1

Question 764:  What are the name , population , and life expectancy of the largest Asian country by land ? ||| world_1
SQL:  select name ,  population ,  lifeexpectancy from country where continent  =  "asia" order by surfacearea desc limit 1

Question 765:  What is average life expectancy in the countries where English is not the official language ? ||| world_1
SQL:  select avg(lifeexpectancy) from country where name not in (select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" and t2.isofficial  =  "t")

Question 766:  Give the mean life expectancy of countries in which English is not the official language . ||| world_1
SQL:  select avg(lifeexpectancy) from country where name not in (select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" and t2.isofficial  =  "t")

Question 767:  What is the total number of people living in the nations that do not use English ? ||| world_1
SQL:  select sum(population) from country where name not in (select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english")

Question 768:  How many people live in countries that do not speak English ? ||| world_1
SQL:  select sum(population) from country where name not in (select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english")

Question 769:  What is the official language spoken in the country whose head of state is Beatrix ? ||| world_1
SQL:  select t2.language from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.headofstate  =  "beatrix" and t2.isofficial  =  "t"

Question 770:  What is the official language used in the country the name of whose head of state is Beatrix . ||| world_1
SQL:  select t2.language from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t1.headofstate  =  "beatrix" and t2.isofficial  =  "t"

Question 771:  What is the total number of unique official languages spoken in the countries that are founded before 1930 ? ||| world_1
SQL:  select count(distinct t2.language) from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where  indepyear  <  1930 and t2.isofficial  =  "t"

Question 772:  For the countries founded before 1930 , what is the total number of distinct official languages ? ||| world_1
SQL:  select count(distinct t2.language) from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where  indepyear  <  1930 and t2.isofficial  =  "t"

Question 773:  What are the countries that have greater surface area than any country in Europe ? ||| world_1
SQL:  select name from country where surfacearea  >  (select min(surfacearea) from country where continent  =  "europe")

Question 774:  Which countries have greater area than that of any country in Europe ? ||| world_1
SQL:  select name from country where surfacearea  >  (select min(surfacearea) from country where continent  =  "europe")

Question 775:  What are the African countries that have a population less than any country in Asia ? ||| world_1
SQL:  select name from country where continent  =  "africa"  and population  <  (select max(population) from country where continent  =  "asia")

Question 776:  Which African countries have a smaller population than that of any country in Asia ? ||| world_1
SQL:  select name from country where continent  =  "africa"  and population  <  (select min(population) from country where continent  =  "asia")

Question 777:  Which Asian countries have a population that is larger than any country in Africa ? ||| world_1
SQL:  select name from country where continent  =  "asia"  and population  >  (select max(population) from country where continent  =  "africa")

Question 778:  What are the Asian countries which have a population larger than that of any country in Africa ? ||| world_1
SQL:  select name from country where continent  =  "asia"  and population  >  (select min(population) from country where continent  =  "africa")

Question 779:  What are the country codes for countries that do not speak English ? ||| world_1
SQL:  select countrycode from countrylanguage except select countrycode from countrylanguage where language  =  "english"

Question 780:  Return the country codes for countries that do not speak English . ||| world_1
SQL:  select countrycode from countrylanguage except select countrycode from countrylanguage where language  =  "english"

Question 781:  What are the country codes of countries where people use languages other than English ? ||| world_1
SQL:  select distinct countrycode from countrylanguage where language != "english"

Question 782:  Give the country codes for countries in which people speak langauges that are not English . ||| world_1
SQL:  select distinct countrycode from countrylanguage where language != "english"

Question 783:  What are the codes of the countries that do not speak English and whose government forms are not Republic ? ||| world_1
SQL:  select code from country where governmentform != "republic" except select countrycode from countrylanguage where language  =  "english"

Question 784:  Return the codes of countries that do not speak English and do not have Republics for governments . ||| world_1
SQL:  select code from country where governmentform != "republic" except select countrycode from countrylanguage where language  =  "english"

Question 785:  Which cities are in European countries where English is not the official language ? ||| world_1
SQL:  select distinct t2.name from country as t1 join city as t2 on t2.countrycode  =  t1.code where t1.continent  =  'europe' and t1.name not in (select t3.name from country as t3 join countrylanguage as t4 on t3.code  =  t4.countrycode where t4.isofficial  =  't' and t4.language  =  'english')

Question 786:  What are the names of cities in Europe for which English is not the official language ? ||| world_1
SQL:  select distinct t2.name from country as t1 join city as t2 on t2.countrycode  =  t1.code where t1.continent  =  'europe' and t1.name not in (select t3.name from country as t3 join countrylanguage as t4 on t3.code  =  t4.countrycode where t4.isofficial  =  't' and t4.language  =  'english')

Question 787:  Which unique cities are in Asian countries where Chinese is the official language ? ||| world_1
SQL:  select distinct t3.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode join city as t3 on t1.code  =  t3.countrycode where t2.isofficial  =  't' and t2.language  =  'chinese' and t1.continent  =  "asia"

Question 788:  Return the different names of cities that are in Asia and for which Chinese is the official language . ||| world_1
SQL:  select distinct t3.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode join city as t3 on t1.code  =  t3.countrycode where t2.isofficial  =  't' and t2.language  =  'chinese' and t1.continent  =  "asia"

Question 789:  What are the name , independence year , and surface area of the country with the smallest population ? ||| world_1
SQL:  select name ,  surfacearea ,  indepyear from country order by population limit 1

Question 790:  Give the name , year of independence , and surface area of the country that has the lowest population . ||| world_1
SQL:  select name ,  surfacearea ,  indepyear from country order by population limit 1

Question 791:  What are the population , name and leader of the country with the largest area ? ||| world_1
SQL:  select name ,  population ,  headofstate from country order by surfacearea desc limit 1

Question 792:  Give the name , population , and head of state for the country that has the largest area . ||| world_1
SQL:  select name ,  population ,  headofstate from country order by surfacearea desc limit 1

Question 793:  Return the country name and the numbers of languages spoken for each country that speaks at least 3 languages . ||| world_1
SQL:  select count(t2.language) ,  t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode group by t1.name having count(*)  >  2

Question 794:  What are the names of countries that speak more than 2 languages , as well as how many languages they speak ? ||| world_1
SQL:  select count(t2.language) ,  t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode group by t1.name having count(*)  >  2

Question 795:  Find the number of cities in each district whose population is greater than the average population of cities ? ||| world_1
SQL:  select count(*) ,  district from city where population  >  (select avg(population) from city) group by district

Question 796:  How many cities in each district have a population that is above the average population across all cities ? ||| world_1
SQL:  select count(*) ,  district from city where population  >  (select avg(population) from city) group by district

Question 797:  Find the government form name and total population for each government form whose average life expectancy is longer than 72 . ||| world_1
SQL:  select sum(population) ,  governmentform from country group by governmentform having avg(lifeexpectancy)  >  72

Question 798:  What are the different government forms and what is the total population of each for government forms that have an average life expectancy greater than 72 ? ||| world_1
SQL:  select sum(population) ,  governmentform from country group by governmentform having avg(lifeexpectancy)  >  72

Question 799:  Find the average life expectancy and total population for each continent where the average life expectancy is shorter than 72 ? ||| world_1
SQL:  select sum(population) ,  avg(lifeexpectancy) ,  continent from country group by continent having avg(lifeexpectancy)  <  72

Question 800:  What are the different continents and the total popuation and average life expectancy corresponding to each , for continents that have an average life expectancy less than 72 ? ||| world_1
SQL:  select sum(population) ,  avg(lifeexpectancy) ,  continent from country group by continent having avg(lifeexpectancy)  <  72

Question 801:  What are the names and areas of countries with the top 5 largest area ? ||| world_1
SQL:  select name ,  surfacearea from country order by surfacearea desc limit 5

Question 802:  Return the names and surface areas of the 5 largest countries . ||| world_1
SQL:  select name ,  surfacearea from country order by surfacearea desc limit 5

Question 803:  What are names of countries with the top 3 largest population ? ||| world_1
SQL:  select name from country order by population desc limit 3

Question 804:  Return the names of the 3 most populated countries . ||| world_1
SQL:  select name from country order by population desc limit 3

Question 805:  What are the names of the nations with the 3 lowest populations ? ||| world_1
SQL:  select name from country order by population asc limit 3

Question 806:  Return the names of the 3 countries with the fewest people . ||| world_1
SQL:  select name from country order by population asc limit 3

Question 807:  how many countries are in Asia ? ||| world_1
SQL:  select count(*) from country where continent  =  "asia"

Question 808:  Count the number of countries in Asia . ||| world_1
SQL:  select count(*) from country where continent  =  "asia"

Question 809:  What are the names of the countries that are in the continent of Europe and have a population of 80000 ? ||| world_1
SQL:  select name from country where continent  =  "europe" and population  =  "80000"

Question 810:  Give the names of countries that are in Europe and have a population equal to 80000 . ||| world_1
SQL:  select name from country where continent  =  "europe" and population  =  "80000"

Question 811:  What is the total population and average area of countries in the continent of North America whose area is bigger than 3000 ? ||| world_1
SQL:  select sum(population) ,  avg(surfacearea) from country where continent  =  "north america" and surfacearea  >  3000

Question 812:  Give the total population and average surface area corresponding to countries in North America that have a surface area greater than 3000 . ||| world_1
SQL:  select sum(population) ,  avg(surfacearea) from country where continent  =  "north america" and surfacearea  >  3000

Question 813:  What are the cities whose population is between 160000 and 900000 ? ||| world_1
SQL:  select name from city where population between 160000 and 900000

Question 814:  Return the names of cities that have a population between 160000 and 900000 . ||| world_1
SQL:  select name from city where population between 160000 and 900000

Question 815:  Which language is spoken by the largest number of countries ? ||| world_1
SQL:  select language from countrylanguage group by language order by count(*) desc limit 1

Question 816:  Give the language that is spoken in the most countries . ||| world_1
SQL:  select language from countrylanguage group by language order by count(*) desc limit 1

Question 817:  What is the language spoken by the largest percentage of people in each country ? ||| world_1
SQL:  select language ,  countrycode ,  max(percentage) from countrylanguage group by countrycode

Question 818:  What are the country codes of the different countries , and what are the languages spoken by the greatest percentage of people for each ? ||| world_1
SQL:  select language ,  countrycode ,  max(percentage) from countrylanguage group by countrycode

Question 819:  What is the total number of countries where Spanish is spoken by the largest percentage of people ? ||| world_1
SQL:  select count(*) ,   max(percentage) from countrylanguage where language  =  "spanish" group by countrycode

Question 820:  Count the number of countries for which Spanish is the predominantly spoken language . ||| world_1
SQL:  select count(*) ,   max(percentage) from countrylanguage where language  =  "spanish" group by countrycode

Question 821:  What are the codes of countries where Spanish is spoken by the largest percentage of people ? ||| world_1
SQL:  select countrycode ,  max(percentage) from countrylanguage where language  =  "spanish" group by countrycode

Question 822:  Return the codes of countries for which Spanish is the predominantly spoken language . ||| world_1
SQL:  select countrycode ,  max(percentage) from countrylanguage where language  =  "spanish" group by countrycode

Question 823:  How many conductors are there ? ||| orchestra
SQL:  select count(*) from conductor

Question 824:  Count the number of conductors . ||| orchestra
SQL:  select count(*) from conductor

Question 825:  List the names of conductors in ascending order of age . ||| orchestra
SQL:  select name from conductor order by age asc

Question 826:  What are the names of conductors , ordered by age ? ||| orchestra
SQL:  select name from conductor order by age asc

Question 827:  What are the names of conductors whose nationalities are not `` USA '' ? ||| orchestra
SQL:  select name from conductor where nationality != 'usa'

Question 828:  Return the names of conductors that do not have the nationality `` USA '' . ||| orchestra
SQL:  select name from conductor where nationality != 'usa'

Question 829:  What are the record companies of orchestras in descending order of years in which they were founded ? ||| orchestra
SQL:  select record_company from orchestra order by year_of_founded desc

Question 830:  Return the record companies of orchestras , sorted descending by the years in which they were founded . ||| orchestra
SQL:  select record_company from orchestra order by year_of_founded desc

Question 831:  What is the average attendance of shows ? ||| orchestra
SQL:  select avg(attendance) from show

Question 832:  Return the average attendance across all shows . ||| orchestra
SQL:  select avg(attendance) from show

Question 833:  What are the maximum and minimum share of performances whose type is not `` Live final '' . ||| orchestra
SQL:  select max(share) ,  min(share) from performance where type != "live final"

Question 834:  Return the maximum and minimum shares for performances that do not have the type `` Live final '' . ||| orchestra
SQL:  select max(share) ,  min(share) from performance where type != "live final"

Question 835:  How many different nationalities do conductors have ? ||| orchestra
SQL:  select count(distinct nationality) from conductor

Question 836:  Count the number of different nationalities of conductors . ||| orchestra
SQL:  select count(distinct nationality) from conductor

Question 837:  List names of conductors in descending order of years of work . ||| orchestra
SQL:  select name from conductor order by year_of_work desc

Question 838:  What are the names of conductors , sorted descending by the number of years they have worked ? ||| orchestra
SQL:  select name from conductor order by year_of_work desc

Question 839:  List the name of the conductor with the most years of work . ||| orchestra
SQL:  select name from conductor order by year_of_work desc limit 1

Question 840:  What is the name of the conductor who has worked the greatest number of years ? ||| orchestra
SQL:  select name from conductor order by year_of_work desc limit 1

Question 841:  Show the names of conductors and the orchestras they have conducted . ||| orchestra
SQL:  select t1.name ,  t2.orchestra from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id

Question 842:  What are the names of conductors as well as the corresonding orchestras that they have conducted ? ||| orchestra
SQL:  select t1.name ,  t2.orchestra from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id

Question 843:  Show the names of conductors that have conducted more than one orchestras . ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id having count(*)  >  1

Question 844:  What are the names of conductors who have conducted at more than one orchestra ? ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id having count(*)  >  1

Question 845:  Show the name of the conductor that has conducted the most number of orchestras . ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id order by count(*) desc limit 1

Question 846:  What is the name of the conductor who has conducted the most orchestras ? ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id order by count(*) desc limit 1

Question 847:  Please show the name of the conductor that has conducted orchestras founded after 2008 . ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id where year_of_founded  >  2008

Question 848:  What are the names of conductors who have conducted orchestras founded after the year 2008 ? ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id where year_of_founded  >  2008

Question 849:  Please show the different record companies and the corresponding number of orchestras . ||| orchestra
SQL:  select record_company ,  count(*) from orchestra group by record_company

Question 850:  How many orchestras does each record company manage ? ||| orchestra
SQL:  select record_company ,  count(*) from orchestra group by record_company

Question 851:  Please show the record formats of orchestras in ascending order of count . ||| orchestra
SQL:  select major_record_format from orchestra group by major_record_format order by count(*) asc

Question 852:  What are the major record formats of orchestras , sorted by their frequency ? ||| orchestra
SQL:  select major_record_format from orchestra group by major_record_format order by count(*) asc

Question 853:  List the record company shared by the most number of orchestras . ||| orchestra
SQL:  select record_company from orchestra group by record_company order by count(*) desc limit 1

Question 854:  What is the record company used by the greatest number of orchestras ? ||| orchestra
SQL:  select record_company from orchestra group by record_company order by count(*) desc limit 1

Question 855:  List the names of orchestras that have no performance . ||| orchestra
SQL:  select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)

Question 856:  What are the orchestras that do not have any performances ? ||| orchestra
SQL:  select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)

Question 857:  Show the record companies shared by orchestras founded before 2003 and after 2003 . ||| orchestra
SQL:  select record_company from orchestra where year_of_founded  <  2003 intersect select record_company from orchestra where year_of_founded  >  2003

Question 858:  What are the record companies that are used by both orchestras founded before 2003 and those founded after 2003 ? ||| orchestra
SQL:  select record_company from orchestra where year_of_founded  <  2003 intersect select record_company from orchestra where year_of_founded  >  2003

Question 859:  Find the number of orchestras whose record format is `` CD '' or `` DVD '' . ||| orchestra
SQL:  select count(*) from orchestra where major_record_format  =  "cd" or major_record_format  =  "dvd"

Question 860:  Count the number of orchestras that have CD or DVD as their record format . ||| orchestra
SQL:  select count(*) from orchestra where major_record_format  =  "cd" or major_record_format  =  "dvd"

Question 861:  Show the years in which orchestras that have given more than one performance are founded . ||| orchestra
SQL:  select year_of_founded from orchestra as t1 join performance as t2 on t1.orchestra_id  =  t2.orchestra_id group by t2.orchestra_id having count(*)  >  1

Question 862:  What are years of founding for orchestras that have had more than a single performance ? ||| orchestra
SQL:  select year_of_founded from orchestra as t1 join performance as t2 on t1.orchestra_id  =  t2.orchestra_id group by t2.orchestra_id having count(*)  >  1

Question 863:  How many high schoolers are there ? ||| network_1
SQL:  select count(*) from highschooler

Question 864:  Count the number of high schoolers . ||| network_1
SQL:  select count(*) from highschooler

Question 865:  Show the names and grades of each high schooler . ||| network_1
SQL:  select name ,  grade from highschooler

Question 866:  What are the names and grades for each high schooler ? ||| network_1
SQL:  select name ,  grade from highschooler

Question 867:  Show all the grades of the high schoolers . ||| network_1
SQL:  select grade from highschooler

Question 868:  What is the grade of each high schooler ? ||| network_1
SQL:  select grade from highschooler

Question 869:  What grade is Kyle in ? ||| network_1
SQL:  select grade from highschooler where name  =  "kyle"

Question 870:  Return the grade for the high schooler named Kyle . ||| network_1
SQL:  select grade from highschooler where name  =  "kyle"

Question 871:  Show the names of all high schoolers in grade 10 . ||| network_1
SQL:  select name from highschooler where grade  =  10

Question 872:  What are the names of all high schoolers in grade 10 ? ||| network_1
SQL:  select name from highschooler where grade  =  10

Question 873:  Show the ID of the high schooler named Kyle . ||| network_1
SQL:  select id from highschooler where name  =  "kyle"

Question 874:  What is Kyle 's id ? ||| network_1
SQL:  select id from highschooler where name  =  "kyle"

Question 875:  How many high schoolers are there in grade 9 or 10 ? ||| network_1
SQL:  select count(*) from highschooler where grade  =  9 or grade  =  10

Question 876:  Count the number of high schoolers in grades 9 or 10 . ||| network_1
SQL:  select count(*) from highschooler where grade  =  9 or grade  =  10

Question 877:  Show the number of high schoolers for each grade . ||| network_1
SQL:  select grade ,  count(*) from highschooler group by grade

Question 878:  How many high schoolers are in each grade ? ||| network_1
SQL:  select grade ,  count(*) from highschooler group by grade

Question 879:  Which grade has the most high schoolers ? ||| network_1
SQL:  select grade from highschooler group by grade order by count(*) desc limit 1

Question 880:  Return the grade that has the greatest number of high schoolers . ||| network_1
SQL:  select grade from highschooler group by grade order by count(*) desc limit 1

Question 881:  Show me all grades that have at least 4 students . ||| network_1
SQL:  select grade from highschooler group by grade having count(*)  >=  4

Question 882:  Which grades have 4 or more high schoolers ? ||| network_1
SQL:  select grade from highschooler group by grade having count(*)  >=  4

Question 883:  Show the student IDs and numbers of friends corresponding to each . ||| network_1
SQL:  select student_id ,  count(*) from friend group by student_id

Question 884:  How many friends does each student have ? ||| network_1
SQL:  select student_id ,  count(*) from friend group by student_id

Question 885:  Show the names of high school students and their corresponding number of friends . ||| network_1
SQL:  select t2.name ,  count(*) from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id

Question 886:  What are the names of the high schoolers and how many friends does each have ? ||| network_1
SQL:  select t2.name ,  count(*) from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id

Question 887:  What is the name of the high schooler who has the greatest number of friends ? ||| network_1
SQL:  select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id order by count(*) desc limit 1

Question 888:  Return the name of the high school student with the most friends . ||| network_1
SQL:  select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id order by count(*) desc limit 1

Question 889:  Show the names of high schoolers who have at least 3 friends . ||| network_1
SQL:  select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id having count(*)  >=  3

Question 890:  What are the names of high schoolers who have 3 or more friends ? ||| network_1
SQL:  select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id having count(*)  >=  3

Question 891:  Show the names of all of the high schooler Kyle 's friends . ||| network_1
SQL:  select t3.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id join highschooler as t3 on t1.friend_id  =  t3.id where t2.name  =  "kyle"

Question 892:  Return the names of friends of the high school student Kyle . ||| network_1
SQL:  select t3.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id join highschooler as t3 on t1.friend_id  =  t3.id where t2.name  =  "kyle"

Question 893:  How many friends does the high school student Kyle have ? ||| network_1
SQL:  select count(*) from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id where t2.name  =  "kyle"

Question 894:  Count the number of friends Kyle has . ||| network_1
SQL:  select count(*) from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id where t2.name  =  "kyle"

Question 895:  Show ids of all students who do not have any friends . ||| network_1
SQL:  select id from highschooler except select student_id from friend

Question 896:  What are the ids of high school students who do not have friends ? ||| network_1
SQL:  select id from highschooler except select student_id from friend

Question 897:  Show names of all high school students who do not have any friends . ||| network_1
SQL:  select name from highschooler except select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id

Question 898:  What are the names of students who have no friends ? ||| network_1
SQL:  select name from highschooler except select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id

Question 899:  Show the ids of high schoolers who have friends and are also liked by someone else . ||| network_1
SQL:  select student_id from friend intersect select liked_id from likes

Question 900:  What are the ids of students who both have friends and are liked ? ||| network_1
SQL:  select student_id from friend intersect select liked_id from likes

Question 901:  Show name of all students who have some friends and also are liked by someone else . ||| network_1
SQL:  select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id intersect select t2.name from likes as t1 join highschooler as t2 on t1.liked_id  =  t2.id

Question 902:  What are the names of high schoolers who both have friends and are liked ? ||| network_1
SQL:  select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id intersect select t2.name from likes as t1 join highschooler as t2 on t1.liked_id  =  t2.id

Question 903:  Count the number of likes for each student id . ||| network_1
SQL:  select student_id ,  count(*) from likes group by student_id

Question 904:  How many likes correspond to each student id ? ||| network_1
SQL:  select student_id ,  count(*) from likes group by student_id

Question 905:  Show the names of high schoolers who have likes , and numbers of likes for each . ||| network_1
SQL:  select t2.name ,  count(*) from likes as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id

Question 906:  What are the names of high schoolers who have likes , and how many likes does each have ? ||| network_1
SQL:  select t2.name ,  count(*) from likes as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id

Question 907:  What is the name of the high schooler who has the greatest number of likes ? ||| network_1
SQL:  select t2.name from likes as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id order by count(*) desc limit 1

Question 908:  Give the name of the student with the most likes . ||| network_1
SQL:  select t2.name from likes as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id order by count(*) desc limit 1

Question 909:  Show the names of students who have at least 2 likes . ||| network_1
SQL:  select t2.name from likes as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id having count(*)  >=  2

Question 910:  What are the names of students who have 2 or more likes ? ||| network_1
SQL:  select t2.name from likes as t1 join highschooler as t2 on t1.student_id  =  t2.id group by t1.student_id having count(*)  >=  2

Question 911:  Show the names of students who have a grade higher than 5 and have at least 2 friends . ||| network_1
SQL:  select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id where t2.grade  >  5 group by t1.student_id having count(*)  >=  2

Question 912:  What are the names of high schoolers who have a grade of over 5 and have 2 or more friends ? ||| network_1
SQL:  select t2.name from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id where t2.grade  >  5 group by t1.student_id having count(*)  >=  2

Question 913:  How many likes does Kyle have ? ||| network_1
SQL:  select count(*) from likes as t1 join highschooler as t2 on t1.student_id  =  t2.id where t2.name  =  "kyle"

Question 914:  Return the number of likes that the high schooler named Kyle has . ||| network_1
SQL:  select count(*) from likes as t1 join highschooler as t2 on t1.student_id  =  t2.id where t2.name  =  "kyle"

Question 915:  Find the average grade of all students who have some friends . ||| network_1
SQL:  select avg(grade) from highschooler where id in (select t1.student_id from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id)

Question 916:  What is the average grade of students who have friends ? ||| network_1
SQL:  select avg(grade) from highschooler where id in (select t1.student_id from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id)

Question 917:  Find the minimum grade of students who have no friends . ||| network_1
SQL:  select min(grade) from highschooler where id not in (select t1.student_id from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id)

Question 918:  What is the lowest grade of students who do not have any friends ? ||| network_1
SQL:  select min(grade) from highschooler where id not in (select t1.student_id from friend as t1 join highschooler as t2 on t1.student_id  =  t2.id)

Question 919:  Which states have both owners and professionals living there ? ||| dog_kennels
SQL:  select state from owners intersect select state from professionals

Question 920:  Find the states where both owners and professionals live . ||| dog_kennels
SQL:  select state from owners intersect select state from professionals

Question 921:  What is the average age of the dogs who have gone through any treatments ? ||| dog_kennels
SQL:  select avg(age) from dogs where dog_id in ( select dog_id from treatments )

Question 922:  Find the average age of the dogs who went through treatments . ||| dog_kennels
SQL:  select avg(age) from dogs where dog_id in ( select dog_id from treatments )

Question 923:  Which professionals live in the state of Indiana or have done treatment on more than 2 treatments ? List his or her id , last name and cell phone . ||| dog_kennels
SQL:  select professional_id ,  last_name ,  cell_number from professionals where state  =  'indiana' union select t1.professional_id ,  t1.last_name ,  t1.cell_number from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id group by t1.professional_id having count(*)  >  2

Question 924:  Find the id , last name and cell phone of the professionals who live in the state of Indiana or have performed more than two treatments . ||| dog_kennels
SQL:  select professional_id ,  last_name ,  cell_number from professionals where state  =  'indiana' union select t1.professional_id ,  t1.last_name ,  t1.cell_number from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id group by t1.professional_id having count(*)  >  2

Question 925:  Which dogs have not cost their owner more than 1000 for treatment ? List the dog names . ||| dog_kennels
SQL:  select name from dogs where dog_id not in ( select dog_id from treatments group by dog_id having sum(cost_of_treatment)  >  1000 )

Question 926:  What are the names of the dogs for which the owner has not spend more than 1000 for treatment ? ||| dog_kennels
SQL:  select name from dogs where dog_id not in ( select dog_id from treatments group by dog_id having sum(cost_of_treatment)  >  1000 )

Question 927:  Which first names are used for professionals or owners but are not used as dog names ? ||| dog_kennels
SQL:  select first_name from professionals union select first_name from owners except select name from dogs

Question 928:  Find the first names that are used for professionals or owners but are not used as dog names . ||| dog_kennels
SQL:  select first_name from professionals union select first_name from owners except select name from dogs

Question 929:  Which professional did not operate any treatment on dogs ? List the professional 's id , role and email . ||| dog_kennels
SQL:  select professional_id ,  role_code ,  email_address from professionals except select t1.professional_id ,  t1.role_code ,  t1.email_address from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id

Question 930:  Give me the id , role and email of the professionals who did not perform any treatment on dogs . ||| dog_kennels
SQL:  select professional_id ,  role_code ,  email_address from professionals except select t1.professional_id ,  t1.role_code ,  t1.email_address from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id

Question 931:  Which owner owns the most dogs ? List the owner id , first name and last name . ||| dog_kennels
SQL:  select t1.owner_id ,  t2.first_name ,  t2.last_name from dogs as t1 join owners as t2 on t1.owner_id  =  t2.owner_id group by t1.owner_id order by count(*) desc limit 1

Question 932:  Return the owner id , first name and last name of the owner who has the most dogs . ||| dog_kennels
SQL:  select t1.owner_id ,  t2.first_name ,  t2.last_name from dogs as t1 join owners as t2 on t1.owner_id  =  t2.owner_id group by t1.owner_id order by count(*) desc limit 1

Question 933:  Which professionals have done at least two treatments ? List the professional 's id , role , and first name . ||| dog_kennels
SQL:  select t1.professional_id ,  t1.role_code ,  t1.first_name from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id group by t1.professional_id having count(*)  >=  2

Question 934:  What are the id , role , and first name of the professionals who have performed two or more treatments ? ||| dog_kennels
SQL:  select t1.professional_id ,  t1.role_code ,  t1.first_name from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id group by t1.professional_id having count(*)  >=  2

Question 935:  What is the name of the breed with the most dogs ? ||| dog_kennels
SQL:  select t1.breed_name from breeds as t1 join dogs as t2 on t1.breed_code  =  t2.breed_code group by t1.breed_name order by count(*) desc limit 1

Question 936:  Which breed do the most dogs have ? Give me the breed name . ||| dog_kennels
SQL:  select t1.breed_name from breeds as t1 join dogs as t2 on t1.breed_code  =  t2.breed_code group by t1.breed_name order by count(*) desc limit 1

Question 937:  Which owner has paid for the most treatments on his or her dogs ? List the owner id and last name . ||| dog_kennels
SQL:  select t1.owner_id ,  t1.last_name from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id join treatments as t3 on t2.dog_id  =  t3.dog_id group by t1.owner_id order by count(*) desc limit 1

Question 938:  Tell me the owner id and last name of the owner who spent the most on treatments of his or her dogs . ||| dog_kennels
SQL:  select t1.owner_id ,  t1.last_name from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id join treatments as t3 on t2.dog_id  =  t3.dog_id group by t1.owner_id order by count(*) desc limit 1

Question 939:  What is the description of the treatment type that costs the least money in total ? ||| dog_kennels
SQL:  select t1.treatment_type_description from treatment_types as t1 join treatments as t2 on t1.treatment_type_code  =  t2.treatment_type_code group by t1.treatment_type_code order by sum(cost_of_treatment) asc limit 1

Question 940:  Give me the description of the treatment type whose total cost is the lowest . ||| dog_kennels
SQL:  select t1.treatment_type_description from treatment_types as t1 join treatments as t2 on t1.treatment_type_code  =  t2.treatment_type_code group by t1.treatment_type_code order by sum(cost_of_treatment) asc limit 1

Question 941:  Which owner has paid the largest amount of money in total for their dogs ? Show the owner id and zip code . ||| dog_kennels
SQL:  select t1.owner_id ,  t1.zip_code from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id join treatments as t3 on t2.dog_id  =  t3.dog_id group by t1.owner_id order by sum(t3.cost_of_treatment) desc limit 1

Question 942:  Find the owner id and zip code of the owner who spent the most money in total for his or her dogs . ||| dog_kennels
SQL:  select t1.owner_id ,  t1.zip_code from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id join treatments as t3 on t2.dog_id  =  t3.dog_id group by t1.owner_id order by sum(t3.cost_of_treatment) desc limit 1

Question 943:  Which professionals have done at least two types of treatments ? List the professional id and cell phone . ||| dog_kennels
SQL:  select t1.professional_id ,  t1.cell_number from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id group by t1.professional_id having count(*)  >=  2

Question 944:  Find the id and cell phone of the professionals who operate two or more types of treatments . ||| dog_kennels
SQL:  select t1.professional_id ,  t1.cell_number from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id group by t1.professional_id having count(*)  >=  2

Question 945:  What are the first name and last name of the professionals who have done treatment with cost below average ? ||| dog_kennels
SQL:  select distinct t1.first_name ,  t1.last_name from professionals as t1 join treatments as t2 where cost_of_treatment  <  ( select avg(cost_of_treatment) from treatments )

Question 946:  Which professionals have operated a treatment that costs less than the average ? Give me theor first names and last names . ||| dog_kennels
SQL:  select distinct t1.first_name ,  t1.last_name from professionals as t1 join treatments as t2 where cost_of_treatment  <  ( select avg(cost_of_treatment) from treatments )

Question 947:  List the date of each treatment , together with the first name of the professional who operated it . ||| dog_kennels
SQL:  select t1.date_of_treatment ,  t2.first_name from treatments as t1 join professionals as t2 on t1.professional_id  =  t2.professional_id

Question 948:  What are the date and the operating professional 's first name of each treatment ? ||| dog_kennels
SQL:  select t1.date_of_treatment ,  t2.first_name from treatments as t1 join professionals as t2 on t1.professional_id  =  t2.professional_id

Question 949:  List the cost of each treatment and the corresponding treatment type description . ||| dog_kennels
SQL:  select t1.cost_of_treatment ,  t2.treatment_type_description from treatments as t1 join treatment_types as t2 on t1.treatment_type_code  =  t2.treatment_type_code

Question 950:  What are the cost and treatment type description of each treatment ? ||| dog_kennels
SQL:  select t1.cost_of_treatment ,  t2.treatment_type_description from treatments as t1 join treatment_types as t2 on t1.treatment_type_code  =  t2.treatment_type_code

Question 951:  List each owner 's first name , last name , and the size of his for her dog . ||| dog_kennels
SQL:  select t1.first_name ,  t1.last_name ,  t2.size_code from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id

Question 952:  What are each owner 's first name , last name , and the size of their dog ? ||| dog_kennels
SQL:  select t1.first_name ,  t1.last_name ,  t2.size_code from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id

Question 953:  List pairs of the owner 's first name and the dogs 's name . ||| dog_kennels
SQL:  select t1.first_name ,  t2.name from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id

Question 954:  What are each owner 's first name and their dogs 's name ? ||| dog_kennels
SQL:  select t1.first_name ,  t2.name from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id

Question 955:  List the names of the dogs of the rarest breed and the treatment dates of them . ||| dog_kennels
SQL:  select t1.name ,  t2.date_of_treatment from dogs as t1 join treatments as t2 on t1.dog_id  =  t2.dog_id where t1.breed_code  =  ( select breed_code from dogs group by breed_code order by count(*) asc limit 1 )

Question 956:  Which dogs are of the rarest breed ? Show their names and treatment dates . ||| dog_kennels
SQL:  select t1.name ,  t2.date_of_treatment from dogs as t1 join treatments as t2 on t1.dog_id  =  t2.dog_id where t1.breed_code  =  ( select breed_code from dogs group by breed_code order by count(*) asc limit 1 )

Question 957:  Which dogs are owned by someone who lives in Virginia ? List the owner 's first name and the dog 's name . ||| dog_kennels
SQL:  select t1.first_name ,  t2.name from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id where t1.state  =  'virginia'

Question 958:  Find the first names of owners living in Virginia and the names of dogs they own . ||| dog_kennels
SQL:  select t1.first_name ,  t2.name from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id where t1.state  =  'virginia'

Question 959:  What are the arriving date and the departing date of the dogs who have gone through a treatment ? ||| dog_kennels
SQL:  select distinct t1.date_arrived ,  t1.date_departed from dogs as t1 join treatments as t2 on t1.dog_id  =  t2.dog_id

Question 960:  Find the arriving date and the departing date of the dogs that received a treatment . ||| dog_kennels
SQL:  select distinct t1.date_arrived ,  t1.date_departed from dogs as t1 join treatments as t2 on t1.dog_id  =  t2.dog_id

Question 961:  List the last name of the owner owning the youngest dog . ||| dog_kennels
SQL:  select t1.last_name from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id where t2.age  =  ( select max(age) from dogs )

Question 962:  Who owns the youngest dog ? Give me his or her last name . ||| dog_kennels
SQL:  select t1.last_name from owners as t1 join dogs as t2 on t1.owner_id  =  t2.owner_id where t2.age  =  ( select max(age) from dogs )

Question 963:  List the emails of the professionals who live in the state of Hawaii or the state of Wisconsin . ||| dog_kennels
SQL:  select email_address from professionals where state  =  'hawaii' or state  =  'wisconsin'

Question 964:  What are the emails of the professionals living in either the state of Hawaii or the state of Wisconsin ? ||| dog_kennels
SQL:  select email_address from professionals where state  =  'hawaii' or state  =  'wisconsin'

Question 965:  What are the arriving date and the departing date of all the dogs ? ||| dog_kennels
SQL:  select date_arrived ,  date_departed from dogs

Question 966:  List the arrival date and the departure date for all the dogs . ||| dog_kennels
SQL:  select date_arrived ,  date_departed from dogs

Question 967:  How many dogs went through any treatments ? ||| dog_kennels
SQL:  select count(distinct dog_id) from treatments

Question 968:  Count the number of dogs that went through a treatment . ||| dog_kennels
SQL:  select count(distinct dog_id) from treatments

Question 969:  How many professionals have performed any treatment to dogs ? ||| dog_kennels
SQL:  select count(distinct professional_id) from treatments

Question 970:  Find the number of professionals who have ever treated dogs . ||| dog_kennels
SQL:  select count(distinct professional_id) from treatments

Question 971:  Which professionals live in a city containing the substring 'West ' ? List his or her role , street , city and state . ||| dog_kennels
SQL:  select role_code ,  street ,  city ,  state from professionals where city like '%west%'

Question 972:  Find the role , street , city and state of the professionals living in a city that contains the substring 'West ' . ||| dog_kennels
SQL:  select role_code ,  street ,  city ,  state from professionals where city like '%west%'

Question 973:  Which owners live in the state whose name contains the substring 'North ' ? List his first name , last name and email . ||| dog_kennels
SQL:  select first_name ,  last_name ,  email_address from owners where state like '%north%'

Question 974:  Return the first name , last name and email of the owners living in a state whose name contains the substring 'North ' . ||| dog_kennels
SQL:  select first_name ,  last_name ,  email_address from owners where state like '%north%'

Question 975:  How many dogs have an age below the average ? ||| dog_kennels
SQL:  select count(*) from dogs where age  <  ( select avg(age) from dogs )

Question 976:  Count the number of dogs of an age below the average . ||| dog_kennels
SQL:  select count(*) from dogs where age  <  ( select avg(age) from dogs )

Question 977:  How much does the most recent treatment cost ? ||| dog_kennels
SQL:  select cost_of_treatment from treatments order by date_of_treatment desc limit 1

Question 978:  Show me the cost of the most recently performed treatment . ||| dog_kennels
SQL:  select cost_of_treatment from treatments order by date_of_treatment desc limit 1

Question 979:  How many dogs have not gone through any treatment ? ||| dog_kennels
SQL:  select count(*) from dogs where dog_id not in ( select dog_id from treatments )

Question 980:  Tell me the number of dogs that have not received any treatment . ||| dog_kennels
SQL:  select count(*) from dogs where dog_id not in ( select dog_id from treatments )

Question 981:  How many owners temporarily do not have any dogs ? ||| dog_kennels
SQL:  select count(*) from owners where owner_id not in ( select owner_id from dogs )

Question 982:  Find the number of owners who do not own any dogs at this moment . ||| dog_kennels
SQL:  select count(*) from owners where owner_id not in ( select owner_id from dogs )

Question 983:  How many professionals did not operate any treatment on dogs ? ||| dog_kennels
SQL:  select count(*) from professionals where professional_id not in ( select professional_id from treatments )

Question 984:  Find the number of professionals who have not treated any dogs . ||| dog_kennels
SQL:  select count(*) from professionals where professional_id not in ( select professional_id from treatments )

Question 985:  List the dog name , age and weight of the dogs who have been abandoned ? 1 stands for yes , and 0 stands for no . ||| dog_kennels
SQL:  select name ,  age ,  weight from dogs where abandoned_yn  =  1

Question 986:  What are the dog name , age and weight of the dogs that were abandoned ? Note that 1 stands for yes , and 0 stands for no in the tables . ||| dog_kennels
SQL:  select name ,  age ,  weight from dogs where abandoned_yn  =  1

Question 987:  What is the average age of all the dogs ? ||| dog_kennels
SQL:  select avg(age) from dogs

Question 988:  Compute the average age of all the dogs . ||| dog_kennels
SQL:  select avg(age) from dogs

Question 989:  What is the age of the oldest dog ? ||| dog_kennels
SQL:  select max(age) from dogs

Question 990:  Tell me the age of the oldest dog . ||| dog_kennels
SQL:  select max(age) from dogs

Question 991:  How much does each charge type costs ? List both charge type and amount . ||| dog_kennels
SQL:  select charge_type ,  charge_amount from charges

Question 992:  List each charge type and its amount . ||| dog_kennels
SQL:  select charge_type ,  charge_amount from charges

Question 993:  How much does the most expensive charge type costs ? ||| dog_kennels
SQL:  select max(charge_amount) from charges

Question 994:  What is the charge amount of the most expensive charge type ? ||| dog_kennels
SQL:  select max(charge_amount) from charges

Question 995:  List the email , cell phone and home phone of all the professionals . ||| dog_kennels
SQL:  select email_address ,  cell_number ,  home_phone from professionals

Question 996:  What are the email , cell phone and home phone of each professional ? ||| dog_kennels
SQL:  select email_address ,  cell_number ,  home_phone from professionals

Question 997:  What are all the possible breed type and size type combinations ? ||| dog_kennels
SQL:  select distinct breed_code ,  size_code from dogs

Question 998:  Find the distinct breed type and size type combinations for dogs . ||| dog_kennels
SQL:  select distinct breed_code ,  size_code from dogs

Question 999:  List the first name of all the professionals along with the description of the treatment they have done . ||| dog_kennels
SQL:  select distinct t1.first_name ,  t3.treatment_type_description from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id join treatment_types as t3 on t2.treatment_type_code  =  t3.treatment_type_code

Question 1000:  What are each professional 's first name and description of the treatment they have performed ? ||| dog_kennels
SQL:  select distinct t1.first_name ,  t3.treatment_type_description from professionals as t1 join treatments as t2 on t1.professional_id  =  t2.professional_id join treatment_types as t3 on t2.treatment_type_code  =  t3.treatment_type_code

Question 1001:  How many singers are there ? ||| singer
SQL:  select count(*) from singer

Question 1002:  What is the count of singers ? ||| singer
SQL:  select count(*) from singer

Question 1003:  List the name of singers in ascending order of net worth . ||| singer
SQL:  select name from singer order by net_worth_millions asc

Question 1004:  What are the names of singers ordered by ascending net worth ? ||| singer
SQL:  select name from singer order by net_worth_millions asc

Question 1005:  What are the birth year and citizenship of singers ? ||| singer
SQL:  select birth_year ,  citizenship from singer

Question 1006:  What are the birth years and citizenships of the singers ? ||| singer
SQL:  select birth_year ,  citizenship from singer

Question 1007:  List the name of singers whose citizenship is not `` France '' . ||| singer
SQL:  select name from singer where citizenship != "france"

Question 1008:  What are the names of the singers who are not French citizens ? ||| singer
SQL:  select name from singer where citizenship != "france"

Question 1009:  Show the name of singers whose birth year is either 1948 or 1949 ? ||| singer
SQL:  select name from singer where birth_year  =  1948 or birth_year  =  1949

Question 1010:  What are the names of the singers whose birth years are either 1948 or 1949 ? ||| singer
SQL:  select name from singer where birth_year  =  1948 or birth_year  =  1949

Question 1011:  What is the name of the singer with the largest net worth ? ||| singer
SQL:  select name from singer order by net_worth_millions desc limit 1

Question 1012:  What is the name of the singer who is worth the most ? ||| singer
SQL:  select name from singer order by net_worth_millions desc limit 1

Question 1013:  Show different citizenship of singers and the number of singers of each citizenship . ||| singer
SQL:  select citizenship ,  count(*) from singer group by citizenship

Question 1014:  For each citizenship , how many singers are from that country ? ||| singer
SQL:  select citizenship ,  count(*) from singer group by citizenship

Question 1015:  Please show the most common citizenship of singers . ||| singer
SQL:  select citizenship from singer group by citizenship order by count(*) desc limit 1

Question 1016:  What is the most common singer citizenship ? ||| singer
SQL:  select citizenship from singer group by citizenship order by count(*) desc limit 1

Question 1017:  Show different citizenships and the maximum net worth of singers of each citizenship . ||| singer
SQL:  select citizenship ,  max(net_worth_millions) from singer group by citizenship

Question 1018:  For each citizenship , what is the maximum net worth ? ||| singer
SQL:  select citizenship ,  max(net_worth_millions) from singer group by citizenship

Question 1019:  Show titles of songs and names of singers . ||| singer
SQL:  select t2.title ,  t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id

Question 1020:  What are the song titles and singer names ? ||| singer
SQL:  select t2.title ,  t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id

Question 1021:  Show distinct names of singers that have songs with sales more than 300000 . ||| singer
SQL:  select distinct t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id where t2.sales  >  300000

Question 1022:  what are the different names of the singers that have sales more than 300000 ? ||| singer
SQL:  select distinct t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id where t2.sales  >  300000

Question 1023:  Show the names of singers that have more than one song . ||| singer
SQL:  select t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name having count(*)  >  1

Question 1024:  What are the names of the singers that have more than one songs ? ||| singer
SQL:  select t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name having count(*)  >  1

Question 1025:  Show the names of singers and the total sales of their songs . ||| singer
SQL:  select t1.name ,  sum(t2.sales) from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name

Question 1026:  For each singer name , what is the total sales for their songs ? ||| singer
SQL:  select t1.name ,  sum(t2.sales) from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name

Question 1027:  List the name of singers that do not have any song . ||| singer
SQL:  select name from singer where singer_id not in (select singer_id from song)

Question 1028:  What is the sname of every sing that does not have any song ? ||| singer
SQL:  select name from singer where singer_id not in (select singer_id from song)

Question 1029:  Show the citizenship shared by singers with birth year before 1945 and after 1955 . ||| singer
SQL:  select citizenship from singer where birth_year  <  1945 intersect select citizenship from singer where birth_year  >  1955

Question 1030:  What are the citizenships that are shared by singers with a birth year before 1945 and after 1955 ? ||| singer
SQL:  select citizenship from singer where birth_year  <  1945 intersect select citizenship from singer where birth_year  >  1955

Question 1031:  How many available features are there in total ? ||| real_estate_properties
SQL:  select count(*) from other_available_features

Question 1032:  What is the feature type name of feature AirCon ? ||| real_estate_properties
SQL:  select t2.feature_type_name from other_available_features as t1 join ref_feature_types as t2 on t1.feature_type_code  =  t2.feature_type_code where t1.feature_name  =  "aircon"

Question 1033:  Show the property type descriptions of properties belonging to that code . ||| real_estate_properties
SQL:  select t2.property_type_description from properties as t1 join ref_property_types as t2 on t1.property_type_code  =  t2.property_type_code group by t1.property_type_code

Question 1034:  What are the names of properties that are either houses or apartments with more than 1 room ? ||| real_estate_properties
SQL:  select property_name from properties where property_type_code  =  "house" union select property_name from properties where property_type_code  =  "apartment" and room_count  >  1
