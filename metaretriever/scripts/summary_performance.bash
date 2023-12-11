#!/usr/bin/env bash
# -*- coding:utf-8 -*-

for record_type in entity relation relation-boundary event record
do

  echo -e "\n==============>" String ${record_type} "<=============="
  python3 scripts/summary_result.py -record ${record_type} -string -model output/* | grep checkpoint-

done

for record_type in entity relation relation-boundary event record
do

  echo -e "\n==============>" Offset ${record_type} "<=============="
  python3 scripts/summary_result.py -record ${record_type} -model output/* | grep checkpoint-
done


for record_type in entity relation relation-boundary event record
do

  echo -e "\n==============>" Mean String ${record_type} "<=============="
  python3 scripts/summary_result.py -mean -reduce run -record ${record_type} -string

  echo -e "\n==============>" String ${record_type} "<=============="
  python3 scripts/summary_result.py -record ${record_type} -string

done

for record_type in entity relation relation-boundary event record
do

  echo -e "\n==============>" Mean Offset ${record_type} "<=============="
  python3 scripts/summary_result.py -mean -reduce run -record ${record_type}

  echo -e "\n==============>" Offset ${record_type} "<=============="
  python3 scripts/summary_result.py -record ${record_type}

done
