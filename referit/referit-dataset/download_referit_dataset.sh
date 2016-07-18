#!/bin/bash
wget -O ./referit/referit-dataset/referitdata.tar.gz http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referitdata.tar.gz
tar -xzvf ./referit/referit-dataset/referitdata.tar.gz -C ./referit/referit-dataset/
wget -O ./referit/data/referit_edgeboxes_top100.zip http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referit_edgeboxes_top100.zip
unzip ./referit/data/referit_edgeboxes_top100.zip -d ./referit/data/
