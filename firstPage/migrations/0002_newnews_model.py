# Generated by Django 3.2.11 on 2022-02-02 08:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('firstPage', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='newnews',
            name='model',
            field=models.CharField(default='Bert', max_length=50),
            preserve_default=False,
        ),
    ]