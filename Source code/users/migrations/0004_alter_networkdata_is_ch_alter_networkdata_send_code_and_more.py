# Generated by Django 5.0.4 on 2024-07-06 12:54

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0003_networkdata_delete_airqualitydata"),
    ]

    operations = [
        migrations.AlterField(
            model_name="networkdata",
            name="is_CH",
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name="networkdata",
            name="send_code",
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name="networkdata",
            name="who_CH",
            field=models.IntegerField(),
        ),
    ]
