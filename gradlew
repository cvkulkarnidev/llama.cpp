#!/bin/sh

APP_HOME=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
CLASSPATH="$APP_HOME/gradle/wrapper/gradle-wrapper.jar"

if [ -n "$JAVA_HOME" ]; then
    JAVACMD="$JAVA_HOME/bin/java"
else
    JAVACMD=java
fi

if ! command -v "$JAVACMD" >/dev/null 2>&1; then
    echo "ERROR: Java was not found. Set JAVA_HOME to a JDK 17 installation." >&2
    exit 1
fi

exec "$JAVACMD" -Dfile.encoding=UTF-8 -Xmx64m -Xms64m \
    -classpath "$CLASSPATH" org.gradle.wrapper.GradleWrapperMain "$@"
