"""merge heads

Revision ID: b50d1d9d329b
Revises: 536aec2b0cc6, a1b2c3d4e5f6
Create Date: 2026-01-02 23:20:24.449958

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b50d1d9d329b'
down_revision: Union[str, None] = ('536aec2b0cc6', 'a1b2c3d4e5f6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
