---
layout: post
title: Open Github repo from terminal
date: 2017-03-22
published: true
categories: tutorials
tags:
---

I was chatting with Andrew this morning, and he had the great idea that it would be awesome to open our GitHub repo pages from the CLI.  We hashed out a rough draft this morning, but I've cleaned it up and bit and thought I would share here.

The following code will open the GH Enterprise repo in the Chrome Web browser when you type the command `gh` in the terminal.  To install, copy the code below into the bottom of your `~/.bash_profile` file in your home directory.  


```bash
# function to open GitHub Enterprise repo of the current working directory
function gh {
    GH_USER_NAME="your_user_name_here"
    REPO_NAME=`git rev-parse --show-toplevel 2>/dev/null`
    if [  $REPO_NAME ]; then
        CLEAN_NAME=`basename $REPO_NAME`
        open -a "Google Chrome.app" "https://git.generalassemb.ly/$GH_USER_NAME/$CLEAN_NAME" 
    else
        echo "This folder does not contain a git repo"
    fi
}
```

This code uses `git rev-parse --show-toplevel` to search for the git info in the current directory.  `2>/dev/null` is just a flag to hide any error messages that might pop up (usually because a repo does not exist).

`if [  $REPO_NAME ]; then` checks if the repo search completed successfully before trying to build and open a link.  If this check fails, a simple error message is displayed without opening the web browser.

`open -a "Google Chrome.app" "https://git.generalassemb.ly/$GH_USER_NAME/$CLEAN_NAME"` opens the link built with the repo name and yout GH username in Chrome.  `-a "Google Chrome.app"` can be deleted to open the URL in your default web browser, or `"Google Chrome.app"` can be changed to the name of whichever browser you prefer for this task (e.g. Firefox, Chromium, or Opera).
