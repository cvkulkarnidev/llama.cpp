@echo off
setlocal

set "APP_HOME=%~dp0"
set "CLASSPATH=%APP_HOME%gradle\wrapper\gradle-wrapper.jar"

if defined JAVA_HOME (
    set "JAVACMD=%JAVA_HOME%\bin\java.exe"
) else (
    set "JAVACMD=java.exe"
)

"%JAVACMD%" -version >NUL 2>&1
if errorlevel 1 (
    echo ERROR: Java was not found. Set JAVA_HOME to a JDK 17 installation. 1>&2
    exit /b 1
)

"%JAVACMD%" -Dfile.encoding=UTF-8 -Xmx64m -Xms64m -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
exit /b %ERRORLEVEL%
