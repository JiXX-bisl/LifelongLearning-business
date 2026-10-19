from __future__ import annotations

import time
from typing import Any

import docker
import mysql.connector


class MySQLEnvironment:
    """Using Docker provide tmp MySQL env"""

    def __init__(self, image:str="mysql:8.0", root_password:str="password"):
        self.image = image
        self.root_password = root_password

        self.docker_client = docker.from_env()

        self.container = self.docker_client.containers.run(
            image=self.image,
            environment={
                "MYSQL_ROOT_PASSWORD": self.root_password
            },
            ports={
                "3306/tcp": None
            },
            detach=True,
            remove=True
        )

        self.container.reload()

        port_binding = self.container.attrs["NetworkSettings"]["Ports"]["3306/tcp"][0]

        self.host_port = int(port_binding["HostPort"])
        self.connection = None

        self._connect()

    def _connect(self) -> None:
        """Waiting for MySql service connection"""
        last_error: Exception | None = None
        for _ in range(30):
            try:
                self.connection = mysql.connector.connect(
                    host="127.0.0.1",
                    port=self.host_port,
                    user="root",
                    password=self.root_password,
                    pool_reset_session=True,
                )
                return
            except mysql.connector.Error as error:
                last_error = error
                time.sleep(2)
        self.container.stop()

        raise RuntimeError(
            "MySql container started, but MySQL did not become ready."
        ) from last_error
    
    def execute(self, sql:str, database:str | None = None) -> list[tuple[Any, ...]]:
        """Execute one or more SQL seperate with ;"""
        if self.connection is None:
            raise RuntimeError("MySQL connection is not available.")
        
        if not self.connection.is_connected():
            self.connection.reconnect(
                attempts=1, delay=0
            )
        cursor = self.connection.cursor()
        result: list[tuple[Any, ...]] = []

        try:
            if database is not None:
                cursor.execute(f"USE `{database}`")
                if cursor.with_rows:
                    cursor.fetchall()
            sql_statement_list = [
                statement.strip()
                for statement in sql.split(";")
                if statement.strip()
            ]

            for statement in sql_statement_list:
                cursor.execute(statement)
                if cursor.with_rows:
                    result = cursor.fetchall()
                else:
                    result = []
            self.connection.commit()
            return result
        finally:
            cursor.close()
        
    def close(self) -> None:
        if (
            self.connection is not None
            and self.connection.is_connected()
        ):
            self.connection.close()
        
        self.container.stop()
