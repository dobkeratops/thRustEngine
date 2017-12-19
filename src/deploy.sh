sudo cp rusteditor.html /Library/WebServer/Documents/
sudo cp shadertset.html /Library/WebServer/Documents/
sudo cp rustv.js /Library/WebServer/Documents/
sudo cp shadertest.js /Library/WebServer/Documents/
sudo cp emgl_ut.js ../../dgithubio

cp rusteditor.html ../../dgithubio
cp shadertest.html ../../dgithubio
cp emgl_ut.js ../../dgithubio
cp main.js ../../dgithubio
cp rustv.js ../../dgithubio
cp shadertest.js ../../dgithubio
pushd ../../dgithubio
git add main.js emgl_ut.js rustv.js shadertest.js shadertest.html rusteditor.html
git commit -am "updated content"
git push origin master
pushd
