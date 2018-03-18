# Commands Cheat Sheet

|**Category**|**Command**|**Description**
-----|-----|-----
|General| |
| |tree -d|show directory structure
| |ls -l \<folder\> \| wc -l|get file count in \<folder\>
| |unzip -q|unzip in quiet mode (doesn't print every unzipped file name)
| |scp -i aws-key-fast-ai.pem \<file to transfer\> ubuntu@\<ip from aws-nb command\>:/home/ubuntu|Transfer a file from local directory on your host machine to remote AWS directory
| |git lfs track *.h5<br/>git add .gitattributes<br/>git add *.h5<br/>git commit -m "Add weights"<br/>git push origin master|git-lfs is a Git extension for versioning large files (like .h5 files of saved weights)<br/><br/>https://git-lfs.github.com/<br/><br/> • How to install on ubuntu: https://askubuntu.com/a/799451<br/>• Git "out of memory, malloc failed" issue: https://stackoverflow.com/a/12672320
|Kaggle CLI| |
| |pip install kaggle-cli|
| |kg config -g -u \<username\> -p \<password\> -c \<competition name\>|
| |kg download|
|AWS CLI| |from sourcing [aws-alias.sh](https://raw.githubusercontent.com/fastai/courses/master/setup/aws-alias.sh)
| |aws-get-p2|Get instance id for P2 instance - must run before other AWS commands will work!
| |aws-ssh|Use sourced aws-alias.sh to quickly access my P2 instance
| |aws-start|Start my P2 instance
| |aws-stop|Stop my P2 instance
| |aws-state|Get state of my P2 instance
| |aws-ip|Get public IP address of my P2 instance
| |aws-nb|Access Jupyter notebook on AWS instance - must run after aws-start
|tmux| |
| |Ctrl B + ?|list of shortcuts
| |Ctrl B + "|split screen horizontally
| |Ctrl B + %|split screen vertically
| |Ctrl B + D|detach from session ("tmux attach" to get back to the session)
| |Ctrl B + arrows|switch screens
| |Ctrl B + [|enter scroll mode ("q" to exit)
|Jupyter| |
| |H|provides keyboard shortcut cheat sheet
| |M|markdown mode
| |Y|code mode
| |Shift + Enter|evaluates cell and provides new cell
| |Esc|pops you back into command mode
| |%|"magic" commands (w/ an autosuggest box)
| |!|bash commands
| |??|see source code
| |Tab|autocomplete a function name
| |Shift + Tab|pops up parameters for a function
| |Shift + Tab, Shift + Tab|pops up documentation for a function
