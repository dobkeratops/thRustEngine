sudo cp rusteditor.html /Library/WebServer/Documents/
pushd ../../dgithubio
cp rusteditor.html ../../dgithubio
cp emgl_ut.js ../../dgithubio
cp main.js ../../dgithubio
git add main.js emgl_ut.js rusteditor.html
git commit -am "updated content"
git push origin master
pushd
