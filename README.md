bundle exec jekyll serve --host 0.0.0.0
git push origin main

GitHub no longer accepts passwords for pushing code - you need to use a Personal Access Token instead. Here's how to fix this:

Step 1: Create a Personal Access Token on GitHub

Go to GitHub.com and log in
Click your profile picture → Settings
Scroll down and click Developer settings (in the left sidebar)
Click Personal access tokens → Tokens (classic)
Click Generate new token → Generate new token (classic)
Give it a name (e.g., "Replit")
Set an expiration date
Check the repo scope (this allows push access)
Click Generate token
Copy the token immediately (you won't see it again!)

git push https://YOUR_USERNAME:YOUR_TOKEN@github.com/beierzhu/beierzhu.github.io.git main