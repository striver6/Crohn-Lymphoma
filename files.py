# -*- coding:utf-8 -*-
import os

os.chdir("D:/")
def list_files(startPath):
    # fileSave = open('list.txt', 'w')
    fileSave = open('list.txt', 'w')
    for root, dirs, files in os.walk(startPath):
        level = root.replace(startPath, '').count(os.sep)
        if(level==0 or level==2):
            print("level:",level)
            indent = '——' * 1 * level
            # fileSave.write('{}{}/'.format(indent, os.path.basename(root)) + '\n')
            # fileSave.write('{}{}\\'.format(indent, os.path.abspath(root)) + '\n')
            subIndent = '——' * 1 * (level + 1)
            for dir in dirs:
                fileSave.write('{}{}\{}'.format(subIndent, os.path.abspath(root), dir) + '\n')
            # for f in files:
                # fileSave.write('{}{}'.format(subIndent, f) + '\n')
                # fileSave.write('{}{}{}'.format(subIndent, os.path.abspath(root), f) + '\n')
    fileSave.close()


dir = 'D:/code'
list_files(dir)