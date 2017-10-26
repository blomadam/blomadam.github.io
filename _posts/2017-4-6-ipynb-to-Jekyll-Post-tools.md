---
layout: post
title: Post Jupyter notebooks to your GitHub blog
date: 2017-04-09
published: true
categories: tutorials
---

Info to help convert the Jupyter notebooks we use in class into blog posts for your github blog.  Manual conversion instructions are given for one-off posts, as well as detials to make a bash alias to automate the conversion process, move the files, and update your blog in one fell swoop.


## Manual instructions

Run the following command in terminal to convert the file named `notebook.ipynb` from Jupyter notebook format to a markdown file suitable for your Jekyll blog.  Note this must be run in the terminal inside the directory that includes the file you intend to convert.

```bash
jupyter nbconvert --to markdown  notebook.ipynb 
```

That command will run the translation utility provided by Jupyter and create two new items:
- `notebook.md` is a new file containing the markdown for your new blog post
- `notebook_files` is a new folder containing all the images for your new blog post


Move the `notebook.md` file to your blog's `_posts` folder and rename it to include the date and title for your post.  You will also want to make sure the first few lines contain the appropriate front matter (YAML block in Jekyll parlance).  You will also need to change any image links to point to your blog's image folder.  Read below to see my tool for automating most of these tasks.

Move the `notebook_files` folder to your blog's "images" folder (this is also automated in my tool below). 

Add these files to the git repository and commit.  Then push the repo up to the server.  Congratulations, your updated site should be live within a few minutes!


## Automated instructions

##### Once you have installed the tool below, the instructions for uploading a notebook as a new post on your blog are:
- Run `new_post notebook.ipynb` (where notebook.ipynb is the name of the file you wish to convert and post)
- Start impatiently refreshing your blog page to see the new post appear.

##### Please note that certain items in the notebook file may not come through to the final blog post:
- LaTeX equations will not be interpreted properly
- Manually inserted images will need to be fixed by hand
- YAML metadata (e.g. title, date, category, etc.) will only work properly if the first cell type is set to **Raw NBConvert**

##### Things my new_post function does:
- Convert to markdown
- Fix inline plot image addresses
- Move post file and images folder to appropriate area of blog directory
- Add and commit new post files to git repo
- Push repo changes to server


There are currently no sanity checks in place in this function, so please take care to understand how it works before you run it, and be careful what you use it on.  I have made an [example post]({% post_url 2017-04-09-Example_post %})


## Automation setup

This process is very similar to the tool I created in the first week of the class.  Basically we are adding a shortcut to our `.bash_profile` script that the CLI loads everytime we open a new terminal window.  I call my shortcut `new_post` but you can change this to whatever command you want to type when you post a notebook to your blog.  The installation is easy, just copy and paste the code below into your `~/.bash_profile` file (you can use the command `subl ~/.bash_profile` to open this file in Sublime Text directly from the CLI).  Do note that you will need to change the info on the first few lines to reflect how your personal computer is setup.

```bash
function new_post {
    #change these 3 lines to match your specific setup
    GH_USER="github_username"
    PC_USER="local_username"
    POST_PATH="/Users/${PC_USER}/dsi-nyc-5/${GH_USER}.github.io/_posts"
    IMG_PATH="/Users/${PC_USER}/dsi-nyc-5/${GH_USER}.github.io/images"

    FILE_NAME="$1"
    CURR_DIR=`pwd`
    FILE_BASE=`basename $FILE_NAME .ipynb`

    POST_NAME="${FILE_BASE}.md"
    IMG_NAME="${FILE_BASE}_files"

    POST_DATE_NAME=`date "+%Y-%m-%d-"`${POST_NAME}

    # convert the notebook
    jupyter nbconvert --to markdown $FILE_NAME

    # change image paths
    sed -i .bak "s:\[png\](:[png](/images/:" $POST_NAME

    # move everything to blog area
    mv  $POST_NAME "${POST_PATH}/${POST_DATE_NAME}"
    mv  $IMG_NAME "${IMG_PATH}/"

    # add files to git repo to be included in next commit
    cd $POST_PATH
    git add $POST_DATE_NAME
    cd $IMG_PATH
    git add $IMG_NAME

    # make git submission
    cd ..
    git commit -m "\"New blog entry ${FILE_BASE}\""

    # push changes to server
    git push

    cd $CURR_DIR
}
```



## Setup pretty tables
When you set up your system for translating the blog posts, you will need to add some CSS code to your html template file to tell Jekyll to format the Pandas DataFrame tables appropraitely.  Below is the relevant section of my `<username>.github.io/_layouts/default.html` file that has the CSS code included at the end of the `<head>` section. You should edit the your defualts file to be similar.

```html
  <head>
    <!-- Begin section for dataframe table formatting -->
    <style type="text/css">
    table.dataframe {
        width: 100%;
        height: 240px;
        display: block;
        overflow: auto;
        font-family: Arial, sans-serif;
        font-size: 13px;
        line-height: 20px;
        text-align: center;
    }
    table.dataframe th {
      font-weight: bold;
      padding: 4px;
    }
    table.dataframe td {
      padding: 4px;
    }
    table.dataframe tr:hover {
      background: #b8d1f3; 
    }
    </style>
    <!-- End section for dataframe table formatting -->
  </head>
  ```

Here is a quick overview of the features in the added CSS code.  You can change any of these settings to suit your fancy.

- Set the table size to be the full width and 240px tall
- Enable scroll bars for accessing larger tables
- Set the font to Arial size 13, and align it to the center of each cell
- Make the table headers (both column and row) bold
- Give each cell 4px of padding
- Highlight the row your mouse hovers over in light blue (`#b8d1f3`)  




